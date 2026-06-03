/*
 * Copyright 2023 Arman Bilge
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.armanbilge.vilcacora.runtime

import cats.effect.{IO, Resource}
import cats.syntax.all._
import com.armanbilge.vilcacora.ir._
import com.armanbilge.vilcacora.runtime.LibSVM._
import scala.scalanative.unsafe._
import scala.scalanative.libc.stdlib
import scala.scalanative.libc.math.{erff, erf, sqrt, sqrtf, tanh, tanhf}
import scala.scalanative.libc.string.memcpy
import scala.scalanative.unsigned._
import scala.collection.mutable.ListBuffer

import com.armanbilge.vilcacora.runtime.BLAS._
import com.armanbilge.vilcacora.runtime.BLASConstants._

/** The core execution engine for a translated `ModelIR`. It manages native memory and executes
  * model operations within a Cats Effect IO context.
  */
object Interpreter {

  /** A type alias mapping a tensor's name to its pointer in native memory. */
  type MemoryMap = Map[String, Ptr[Byte]]

  /** Executes a complete ModelIR graph.
    *
    * The process is separated into two stages:
    *   1. A synchronous validation of the model to fail-fast on unsupported operations. 2. An
    *      asynchronous, resource-safe execution of the graph operations within an IO context.
    *
    * @param model
    *   The intermediate representation of the model to execute.
    * @param inputs
    *   A map of input tensor names to their corresponding Scala arrays.
    * @return
    *   An IO containing a map of output tensor names to their resulting Scala arrays.
    */
  def execute(
      model: ModelIR,
      inputs: Map[String, Array[_]],
  ): Resource[IO, IO[Map[String, Array[_]]]] = {
    validateModel(model)
    val outputArrays: Map[String, Array[_]] = createOutputArrays(model)

    memoryResource(model, inputs, outputArrays).flatMap { memoryMap =>
      val opResources: List[Resource[IO, IO[Unit]]] =
        model.operations.map(op => executeOperation(op, memoryMap, model))

      val combined: Resource[IO, List[IO[Unit]]] = opResources.sequence

      combined.map { opIOs =>
        val runOps: IO[Unit] = opIOs.traverse_(_ *> IO.cede)
        runOps.as(outputArrays)
      }
    }
  }

  /** Synchronously validates the model definition, throwing a `NotImplementedError` if any
    * operation or data type cast is not supported. This ensures the interpreter fails before any
    * memory is allocated or side effects are scheduled.
    */
  private def validateModel(model: ModelIR): Unit = {

    def validateBroadcastOp(inputs: List[String], outputs: List[String], opName: String): Unit = {
      val shapeA = model.allocations(inputs(0)).shape
      val shapeB = model.allocations(inputs(1)).shape
      val outputShape = model.allocations(outputs.head).shape
      val allocA = model.allocations(inputs(0))
      val allocB = model.allocations(inputs(1))
      val allocOut = model.allocations(outputs.head)

      require(
        allocA.dataType == allocB.dataType && allocB.dataType == allocOut.dataType,
        s"$opName requires all tensors to have the same data type. " +
          s"Got: ${allocA.dataType}, ${allocB.dataType}, ${allocOut.dataType}",
      )

      val broadcastedShape = calculateBroadcastShape(shapeA, shapeB)
      require(
        broadcastedShape.isDefined,
        s"$opName inputs are not broadcast compatible: " +
          s"${shapeA.mkString("x")} and ${shapeB.mkString("x")}",
      )
      require(
        broadcastedShape.contains(outputShape),
        s"$opName output shape mismatch: " +
          s"broadcast of inputs gives ${broadcastedShape.get.mkString("x")} " +
          s"but output is ${outputShape.mkString("x")}",
      )
    }

    model.operations.foreach {
      case _: Operation.SVMClassifier => // softmax currently here only because axis is ignored
        () // Supported
      case op: Operation.Add => validateBroadcastOp(op.inputs, op.outputs, "Add")
      case op: Operation.Mul => validateBroadcastOp(op.inputs, op.outputs, "Mul")
      case op: Operation.Div => validateBroadcastOp(op.inputs, op.outputs, "Div")
      case op: Operation.And =>
        validateBroadcastOp(op.inputs, op.outputs, "And")
        (
          model.allocations(op.inputA).dataType,
          model.allocations(op.inputB).dataType,
          model.allocations(op.output).dataType,
        ) match {
          case (DataType.Bool, DataType.Bool, DataType.Bool) => ()
          case (a, b, c) =>
            throw new IllegalStateException(
              s"And is only possible for boolean , got $a $b and $c instead",
            )
        }
      case op: Operation.Where =>
        val allocCond = model.allocations(op.condition)
        val allocA = model.allocations(op.inputA)
        val allocB = model.allocations(op.inputB)
        val allocOut = model.allocations(op.output)

        // 1. Validate Data Types
        require(
          allocCond.dataType == DataType.Bool,
          s"Where condition must be a boolean tensor. Got: ${allocCond.dataType}",
        )
        require(
          allocA.dataType == allocB.dataType && allocB.dataType == allocOut.dataType,
          s"Where requires inputA, inputB, and output to have the same data type. " +
            s"Got: A=${allocA.dataType}, B=${allocB.dataType}, Out=${allocOut.dataType}",
        )

        // 2. Validate 3-Way Broadcasting
        val shapeCond = allocCond.shape
        val shapeA = allocA.shape
        val shapeB = allocB.shape
        val outputShape = allocOut.shape

        // Chain the broadcast calculations: (A broadcast B) broadcast Cond
        val broadcastedShape = calculateBroadcastShape(shapeA, shapeB)
          .flatMap(intermediateShape => calculateBroadcastShape(intermediateShape, shapeCond))

        require(
          broadcastedShape.isDefined,
          s"Where inputs are not broadcast compatible: " +
            s"Cond=$shapeCond, A=$shapeA, B=$shapeB",
        )

        require(
          broadcastedShape.contains(outputShape),
          s"Where output shape mismatch: " +
            s"3-way broadcast gives ${broadcastedShape.get.mkString("x")} " +
            s"but output is ${outputShape.mkString("x")}",
        )
      case op: Operation.Softmax =>
        val inputAlloc = model.allocations(op.input)
        val rank = inputAlloc.shape.length
        require(
          op.axis >= -rank && op.axis < rank,
          s"Softmax axis must be in range [-$rank, ${rank - 1}], got: ${op.axis}",
        )
      case op: Operation.Erf =>
        val input = model.allocations(op.input)
        val output = model.allocations(op.output)
        require(
          input.shape == output.shape && input.dataType == output.dataType,
          s"Erf output should have same shape and type as input",
        )
        input.dataType match {
          case DataType.Float32 | DataType.Float64 =>
            () // Float16/BFloat16 removed until handler supports them
          case typ =>
            throw new IllegalStateException(s"Erf works only for Float32/Float64, got $typ")
        }
      case op: Operation.Tanh =>
        val input = model.allocations(op.input)
        val output = model.allocations(op.output)
        require(
          input.shape == output.shape && input.dataType == output.dataType,
          s"Tanh output should have same shape and type as input",
        )
        input.dataType match {
          case DataType.Float32 | DataType.Float64 =>
            () // Float16/BFloat16 removed until handler supports them
          case typ =>
            throw new IllegalStateException(s"Tanh works only for Float32/Float64, got $typ")
        }
      case op: Operation.Cast =>
        val from = model.allocations(op.input).dataType
        val to = model.allocations(op.output).dataType
        (from, to) match {
          case (f, t) if f == t => ()
          case (DataType.Float64, DataType.Float32) => ()
          case (DataType.Float32, DataType.Float64) => ()
          case (DataType.Int32, DataType.Bool) | (DataType.Int64, DataType.Bool) => ()
          case (from, to) =>
            throw new NotImplementedError(s"Cast from $from to $to is not implemented.")
        }
      case op: Operation.Relu =>
        // Validate ReLU operation requirements
        val inputAlloc = model.allocations(op.input)
        inputAlloc.dataType match {
          case DataType.Float32 | DataType.Float64 => () // Supported
          case unsupported =>
            throw new NotImplementedError(
              s"ReLU operation not implemented for data type: $unsupported",
            )
        }
      case op: Operation.Reshape =>
        // Validate reshape operation requirements
        val inputAlloc = model.allocations(op.input)
        val outputAlloc = model.allocations(op.output)
        require(
          inputAlloc.shape.product == outputAlloc.shape.product,
          s"Reshape operation '${op.input} -> ${op.output}' requires same total elements. " +
            s"Input shape ${inputAlloc.shape} has ${inputAlloc.shape.product} elements, " +
            s"output shape ${outputAlloc.shape} has ${outputAlloc.shape.product} elements",
        )

      case op: Operation.MatMul =>
        // Validate MatMul operation requirements (ONNX Spec)
        val inputAAlloc = model.allocations(op.inputA)
        val inputBAlloc = model.allocations(op.inputB)
        val outputAlloc = model.allocations(op.output)

        require(
          inputAAlloc.dataType == inputBAlloc.dataType &&
            inputBAlloc.dataType == outputAlloc.dataType,
          s"MatMul operation requires all tensors to have the same data type. " +
            s"Got: ${inputAAlloc.dataType}, ${inputBAlloc.dataType}, ${outputAlloc.dataType}",
        )

        inputAAlloc.dataType match {
          case DataType.Float32 | DataType.Float64 => () // Supported
          case unsupported =>
            throw new NotImplementedError(
              s"MatMul operation not implemented for data type: $unsupported",
            )
        }

        // 1-D Promotion per ONNX spec
        val shapeA =
          if (inputAAlloc.shape.length == 1) 1 :: inputAAlloc.shape else inputAAlloc.shape
        val shapeB =
          if (inputBAlloc.shape.length == 1) inputBAlloc.shape :+ 1 else inputBAlloc.shape

        require(
          shapeA.length >= 2 && shapeB.length >= 2,
          s"MatMul requires at least 1-D inputs. Got native shapes: ${inputAAlloc.shape}, ${inputBAlloc.shape}",
        )

        // Matrix inner dimension check
        val k_a = shapeA.last
        val k_b = shapeB(shapeB.length - 2)
        require(
          k_a == k_b,
          s"MatMul K mismatch: A has $k_a cols, B has $k_b rows (after 1-D promotion)",
        )

        // Batch Broadcasting Validation
        val batchA = shapeA.dropRight(2)
        val batchB = shapeB.dropRight(2)
        val batchRank = batchA.length.max(batchB.length)

        val paddedA = List.fill(batchRank - batchA.length)(1) ++ batchA
        val paddedB = List.fill(batchRank - batchB.length)(1) ++ batchB

        val batchOut = paddedA.zip(paddedB).map { case (a, b) =>
          require(
            a == b || a == 1 || b == 1,
            s"MatMul batch dims not broadcastable: $a vs $b",
          )
          a.max(b)
        }

        // Validate Final Output Shape
        val m = shapeA(shapeA.length - 2)
        val n = shapeB.last
        val expectedOutputShape = batchOut ++ List(m, n)

        require(
          outputAlloc.shape == expectedOutputShape,
          s"MatMul output shape mismatch: expected $expectedOutputShape, got ${outputAlloc.shape}",
        )
      case op: Operation.Gemm =>
        val inputAAlloc = model.allocations(op.inputA)
        val inputBAlloc = model.allocations(op.inputB)
        val outputAlloc = model.allocations(op.output)

        require(
          inputAAlloc.dataType == inputBAlloc.dataType &&
            inputBAlloc.dataType == outputAlloc.dataType,
          s"Gemm requires A, B, and Output to have the same data type. Got: ${inputAAlloc.dataType}",
        )

        op.inputC.foreach { c =>
          require(
            model.allocations(c).dataType == outputAlloc.dataType,
            s"Gemm input C must match data type. Got: ${model.allocations(c).dataType}",
          )
        }

        //  Validate Matrix Dimensions considering transpositions
        val shapeA = inputAAlloc.shape
        val shapeB = inputBAlloc.shape
        val shapeOut = outputAlloc.shape

        require(shapeA.length == 2 && shapeB.length == 2, "Gemm requires A and B to be 2D matrices")

        val m = if (op.transA != 0) shapeA(1) else shapeA(0)
        val k_a = if (op.transA != 0) shapeA(0) else shapeA(1)
        val k_b = if (op.transB != 0) shapeB(1) else shapeB(0)
        val n = if (op.transB != 0) shapeB(0) else shapeB(1)

        require(k_a == k_b, s"Gemm inner dimension mismatch: A inner=$k_a, B inner=$k_b")
        require(
          shapeOut == List(m, n),
          s"Gemm output shape mismatch: expected [$m, $n], got $shapeOut",
        )

        //  Validate C Broadcastin
        op.inputC.foreach { c =>
          val shapeC = model.allocations(c).shape
          val broadcastedShape = calculateBroadcastShape(shapeC, List(m, n))
          require(
            broadcastedShape.contains(List(m, n)),
            s"Gemm input C shape $shapeC cannot be broadcast to output shape [$m, $n]",
          )
        }

        inputAAlloc.dataType match {
          case DataType.Float32 | DataType.Float64 => ()
          case unsupported =>
            throw new NotImplementedError(s"Gemm not implemented for: $unsupported")
        }
      case op: Operation.Conv =>
        // Validate Conv operation for MLPack compatibility
        val inputAlloc = model.allocations(op.input)

        // MLPack only supports Float32 tensors (converted to Float64 internally)
        inputAlloc.dataType match {
          case DataType.Float32 | DataType.Float64 => () // Supported
          case unsupported =>
            throw new NotImplementedError(
              s"Conv operation with MLPack only supports Float32 or Float64, got: $unsupported",
            )
        }

        // MLPack convolution limitations
        val padsAreZero = op.pads.forall(_ == 0)
        val autoPadOk = op.autoPad match {
          case AutoPad.SameUpper | AutoPad.Valid => true
          case AutoPad.NotSet if padsAreZero => true // treat as VALID
          case _ => false
        }
        require(
          autoPadOk,
          s"MLPack Conv supports SAME_UPPER, VALID, or NOTSET with zero pads; " +
            s"got autoPad=${op.autoPad}, pads=${op.pads}",
        )

        require(
          op.group == 1,
          s"MLPack Conv does not support grouping (group=${op.group}), only group=1",
        )

        require(
          op.dilations.forall(_ == 1),
          s"MLPack Conv does not support dilation (dilations=${op.dilations}), only [1,1]",
        )

      case op: Operation.MaxPool =>
        // Validate MaxPool operation for MLPack compatibility
        val inputAlloc = model.allocations(op.input)

        inputAlloc.dataType match {
          case DataType.Float32 | DataType.Float64 => () // Supported
          case unsupported =>
            throw new NotImplementedError(
              s"MaxPool operation with MLPack only supports Float32 or Float64, got: $unsupported",
            )
        }

        // MLPack does not implement dilation or ceil_mode
        require(op.dilations.forall(_ == 1), "MLPack MaxPool requires dilations = [1,1]")
        require(!op.ceilMode, "MLPack MaxPool does not support ceil_mode = true")

      case op: Operation.Gather =>
        val inputAlloc = model.allocations(op.input)
        val indicesAlloc = model.allocations(op.indices)
        val rank = inputAlloc.shape.length
        indicesAlloc.dataType match {
          case DataType.Int32 | DataType.Int64 => () // Supported
          case unsupported =>
            throw new IllegalStateException(
              s"Indices tensor must have data type Int32 or Int64, got: $unsupported",
            )
        }
        require(
          op.axis >= -rank && op.axis < rank,
          s"axis must be in range [-$rank, ${rank - 1}], got: ${op.axis}",
        )
      case op: Operation.GatherND =>
        val inputAlloc = model.allocations(op.input)
        val indicesAlloc = model.allocations(op.indices)
        val dataShape = inputAlloc.shape
        val indicesShape = indicesAlloc.shape
        val dataRank = dataShape.length
        val indicesRank = indicesShape.length
        val indexDepth = indicesShape.last

        indicesAlloc.dataType match {
          case DataType.Int32 | DataType.Int64 => ()
          case unsupported =>
            throw new IllegalStateException(
              s"GatherND: indices must be Int32 or Int64, got: $unsupported",
            )
        }

        require(
          op.batchDims >= 0,
          s"GatherND: batchDims must be >= 0, got: ${op.batchDims}",
        )

        require(
          op.batchDims < dataRank,
          s"GatherND: batchDims (${op.batchDims}) must be < data rank ($dataRank)",
        )
        require(
          op.batchDims < indicesRank,
          s"GatherND: batchDims (${op.batchDims}) must be < indices rank ($indicesRank)",
        )

        require(
          op.batchDims + indexDepth <= dataRank,
          s"GatherND: batchDims (${op.batchDims}) + indexDepth ($indexDepth) " +
            s"must be <= data rank ($dataRank)",
        )

        if (op.batchDims > 0) {
          val dataBatchDims = dataShape.take(op.batchDims)
          val indicesBatchDims = indicesShape.take(op.batchDims)
          require(
            dataBatchDims == indicesBatchDims,
            s"GatherND: batch dimensions of data and indices must match. " +
              s"data batch dims: $dataBatchDims, indices batch dims: $indicesBatchDims",
          )
        }
      case op: Operation.IsNaN =>
        val input = model.allocations(op.input)
        val output = model.allocations(op.output)

        require(input.shape == output.shape, s"input and output tensors should have same shape")

        input.dataType match {
          case DataType.Float32 | DataType.Float64 => ()
          case _ =>
            throw new IllegalStateException(s"isNaN works only for Float32 and Float64 inputs")
        }
        output.dataType match {
          case DataType.Bool => ()
          case _ => throw new IllegalStateException(s"isNaN output should only be boolean")
        }
      case op: Operation.Transpose =>
        val inputAlloc = model.allocations(op.input)
        val outputAlloc = model.allocations(op.output)
        val rank = inputAlloc.shape.length

        require(
          inputAlloc.dataType == outputAlloc.dataType,
          s"Transpose output must have same data type as input. Got: in=${inputAlloc.dataType}, out=${outputAlloc.dataType}",
        )

        val actualPerm = if (op.perm.isEmpty) (rank - 1 to 0 by -1).toList else op.perm

        require(
          actualPerm.length == rank,
          s"Transpose perm length (${actualPerm.length}) must equal input rank ($rank)",
        )

        val isPermutation = actualPerm.sorted == (0 until rank).toList
        require(
          isPermutation,
          s"Transpose perm must be a valid permutation of 0 to ${rank - 1}. Got: $actualPerm",
        )

        val expectedOutputShape = actualPerm.map(i => inputAlloc.shape(i))
        require(
          outputAlloc.shape == expectedOutputShape,
          s"Transpose output shape mismatch. Expected: $expectedOutputShape, Got: ${outputAlloc.shape}",
        )
      case op: Operation.LayerNormalization =>
        val inputAlloc = model.allocations(op.input)
        val scaleAlloc = model.allocations(op.scale)
        val outputAlloc = model.allocations(op.output)
        val inputShape = inputAlloc.shape
        val rank = inputShape.length

        inputAlloc.dataType match {
          case DataType.Float32 | DataType.Float64 => ()
          case unsupported =>
            throw new NotImplementedError(
              s"LayerNormalization not implemented for data type: $unsupported",
            )
        }

        require(
          inputAlloc.dataType == scaleAlloc.dataType &&
            scaleAlloc.dataType == outputAlloc.dataType,
          s"LayerNormalization requires input, scale, and output to share dtype. " +
            s"Got: input=${inputAlloc.dataType}, scale=${scaleAlloc.dataType}, out=${outputAlloc.dataType}",
        )

        val axis = if (op.axis < 0) op.axis + rank else op.axis
        require(
          axis >= 0 && axis < rank,
          s"LayerNormalization axis ${op.axis} is out of range for rank $rank",
        )

        val normShape = inputShape.drop(axis)
        require(
          scaleAlloc.shape == normShape,
          s"LayerNormalization scale shape ${scaleAlloc.shape} must match " +
            s"normalised dims $normShape",
        )
        op.bias.foreach { b =>
          val biasAlloc = model.allocations(b)
          require(
            biasAlloc.shape == normShape,
            s"LayerNormalization bias shape ${biasAlloc.shape} must match " +
              s"normalised dims $normShape",
          )
          require(
            biasAlloc.dataType == inputAlloc.dataType,
            s"LayerNormalization bias dtype ${biasAlloc.dataType} must match input dtype",
          )
        }

        require(
          outputAlloc.shape == inputShape,
          s"LayerNormalization output shape ${outputAlloc.shape} must match input shape $inputShape",
        )

        val outerShape = if (axis == 0) List(1) else inputShape.take(axis)
        op.mean.foreach { m =>
          val meanAlloc = model.allocations(m)
          require(
            meanAlloc.shape == outerShape,
            s"LayerNormalization mean output shape ${meanAlloc.shape} must be $outerShape",
          )
        }
        op.inverseStdDeviation.foreach { s =>
          val invAlloc = model.allocations(s)
          require(
            invAlloc.shape == outerShape,
            s"LayerNormalization invStdDev output shape ${invAlloc.shape} must be $outerShape",
          )
        }
      case other =>
        throw new NotImplementedError(s"Operation not implemented: ${other.getClass.getSimpleName}")
    }
  }

  /** A `Resource` that manages all memory for the graph execution.
    *   - Input and output tensors get direct pointers to the memory of their Scala arrays
    *     (zero-copy).
    *   - Intermediate and constant tensors are allocated in native memory using `malloc`. The
    *     `Resource` guarantees that all `malloc`'d memory is freed after execution.
    */
  private def memoryResource(
      model: ModelIR,
      inputs: Map[String, Array[_]],
      outputs: Map[String, Array[_]],
  ): Resource[IO, MemoryMap] = {
    val acquire = IO {
      val mallocedPtrs = ListBuffer.empty[Ptr[Byte]]
      val memoryMap = model.allocations.map { case (name, allocation) =>
        val ptr: Ptr[Byte] =
          if (inputs.contains(name)) {
            inputs(name).at(0).asInstanceOf[Ptr[Byte]]
          } else if (outputs.contains(name)) {
            outputs(name).at(0).asInstanceOf[Ptr[Byte]]
          } else {
            val totalBytes = (allocation.shape.product * allocation.dataType.sizeInBytes).toUSize
            val p = stdlib.malloc(totalBytes)
            if (p == null) throw new OutOfMemoryError(s"Failed to allocate tensor '$name'")

            allocation.initialData.foreach(data =>
              memcpy(p, data.at(0).asInstanceOf[Ptr[Byte]], data.length.toUSize),
            )
            mallocedPtrs += p
            p
          }
        name -> ptr
      }
      (memoryMap.toMap, mallocedPtrs.toList)
    }

    Resource
      .make(acquire) { case (_, ptrsToFree) =>
        IO(ptrsToFree.foreach(stdlib.free))
      }
      .map(_._1)
  }

  /** Dispatches a single operation to its corresponding handler function. */
  private def executeOperation(
      op: Operation,
      memory: MemoryMap,
      model: ModelIR,
  ): Resource[IO, IO[Unit]] =
    op match {
      case op: Operation.SVMClassifier => handleSvmClassifier(op, memory, model)
      case op: Operation.Add => Resource.pure(handleAdd(op, memory, model))
      case op: Operation.Mul => Resource.pure(handleMul(op, memory, model))
      case op: Operation.Cast => Resource.pure(handleCast(op, memory, model))
      case op: Operation.Relu => Resource.pure(handleRelu(op, memory, model))
      case op: Operation.Reshape => Resource.pure(handleReshape(op, memory, model))
      case op: Operation.Conv => handleConv(op, memory, model)
      case op: Operation.MaxPool => handleMaxPool(op, memory, model)
      case op: Operation.MatMul => Resource.pure(handleMatMul(op, memory, model))
      case op: Operation.Softmax => Resource.pure(handleSoftmax(op, memory, model))
      case op: Operation.Gather => Resource.pure(handleGather(op, memory, model))
      case op: Operation.GatherND => Resource.pure(handleGatherND(op, memory, model))
      case op: Operation.Gemm => Resource.pure(handleGemm(op, memory, model))
      case op: Operation.IsNaN => Resource.pure(handleIsNaN(op, memory, model))
      case op: Operation.And => Resource.pure(handleAnd(op, memory, model))
      case op: Operation.Div => Resource.pure(handleDiv(op, memory, model))
      case op: Operation.Erf => Resource.pure(handleErf(op, memory, model))
      case op: Operation.Tanh => Resource.pure(handleTanh(op, memory, model))
      case op: Operation.Where => Resource.pure(handleWhere(op, memory, model))
      case op: Operation.Transpose => Resource.pure(handleTranspose(op, memory, model))
      case op: Operation.LayerNormalization =>
        Resource.pure(handleLayerNormalization(op, memory, model))
      case other =>
        // This case is unreachable due to the pre-validation step.
        // It remains as a safeguard against internal logic errors.
        throw new NotImplementedError(s"Operation not implemented: ${other.getClass.getSimpleName}")
    }

  /** Handles element-wise addition for both Float32 and Float64 tensors. */
  private def handleAdd(op: Operation.Add, memory: MemoryMap, model: ModelIR): IO[Unit] = IO {
    val shapeA = model.allocations(op.inputA).shape
    val shapeB = model.allocations(op.inputB).shape
    val outputShape = model.allocations(op.output).shape
    val dataType = model.allocations(op.inputA).dataType

    if (shapeA == shapeB) {
      // Fast path: same shapes, simple element-wise addition
      val count = outputShape.product
      dataType match {
        case DataType.Float32 =>
          val inputA = memory(op.inputA).asInstanceOf[Ptr[CFloat]]
          val inputB = memory(op.inputB).asInstanceOf[Ptr[CFloat]]
          val output = memory(op.output).asInstanceOf[Ptr[CFloat]]
          var i = 0
          while (i < count) {
            !(output + i) = !(inputA + i) + !(inputB + i)
            i += 1
          }

        case DataType.Float64 =>
          val inputA = memory(op.inputA).asInstanceOf[Ptr[CDouble]]
          val inputB = memory(op.inputB).asInstanceOf[Ptr[CDouble]]
          val output = memory(op.output).asInstanceOf[Ptr[CDouble]]
          var i = 0
          while (i < count) {
            !(output + i) = !(inputA + i) + !(inputB + i)
            i += 1
          }

        case DataType.Int32 =>
          val inputA = memory(op.inputA).asInstanceOf[Ptr[CInt]]
          val inputB = memory(op.inputB).asInstanceOf[Ptr[CInt]]
          val output = memory(op.output).asInstanceOf[Ptr[CInt]]
          var i = 0
          while (i < count) {
            !(output + i) = !(inputA + i) + !(inputB + i)
            i += 1
          }

        case DataType.Int64 =>
          val inputA = memory(op.inputA).asInstanceOf[Ptr[CLongLong]]
          val inputB = memory(op.inputB).asInstanceOf[Ptr[CLongLong]]
          val output = memory(op.output).asInstanceOf[Ptr[CLongLong]]
          var i = 0
          while (i < count) {
            !(output + i) = !(inputA + i) + !(inputB + i)
            i += 1
          }

        case unsupported =>
          throw new NotImplementedError(
            s"Add operation not implemented for data type: $unsupported",
          )
      }
    } else {
      // Broadcasting path: different shapes
      val outArr = outputShape.toArray
      val stA = computeBroadcastStrides(shapeA, outArr.length)
      val stB = computeBroadcastStrides(shapeB, outArr.length)

      var outFlat = 0
      dataType match {
        case DataType.Float32 =>
          val a = memory(op.inputA).asInstanceOf[Ptr[CFloat]]
          val b = memory(op.inputB).asInstanceOf[Ptr[CFloat]]
          val o = memory(op.output).asInstanceOf[Ptr[CFloat]]
          broadcastLoopN(outArr, Array(stA, stB)) { idx =>
            !(o + outFlat) = !(a + idx(0)) + !(b + idx(1))
            outFlat += 1
          }

        case DataType.Float64 =>
          val a = memory(op.inputA).asInstanceOf[Ptr[CDouble]]
          val b = memory(op.inputB).asInstanceOf[Ptr[CDouble]]
          val o = memory(op.output).asInstanceOf[Ptr[CDouble]]
          broadcastLoopN(outArr, Array(stA, stB)) { idx =>
            !(o + outFlat) = !(a + idx(0)) + !(b + idx(1))
            outFlat += 1
          }

        case DataType.Int32 =>
          val a = memory(op.inputA).asInstanceOf[Ptr[CInt]]
          val b = memory(op.inputB).asInstanceOf[Ptr[CInt]]
          val o = memory(op.output).asInstanceOf[Ptr[CInt]]
          broadcastLoopN(outArr, Array(stA, stB)) { idx =>
            !(o + outFlat) = !(a + idx(0)) + !(b + idx(1))
            outFlat += 1
          }

        case DataType.Int64 =>
          val a = memory(op.inputA).asInstanceOf[Ptr[CLongLong]]
          val b = memory(op.inputB).asInstanceOf[Ptr[CLongLong]]
          val o = memory(op.output).asInstanceOf[Ptr[CLongLong]]
          broadcastLoopN(outArr, Array(stA, stB)) { idx =>
            !(o + outFlat) = !(a + idx(0)) + !(b + idx(1))
            outFlat += 1
          }

        case unsupported =>
          throw new NotImplementedError(
            s"Broadcast Add not implemented for data type: $unsupported",
          )
      }
    }
  }

  /** Handles element-wise multiplication for both Float32 and Float64 tensors. */
  private def handleMul(op: Operation.Mul, memory: MemoryMap, model: ModelIR): IO[Unit] = IO {
    val shapeA = model.allocations(op.inputA).shape
    val shapeB = model.allocations(op.inputB).shape
    val outputShape = model.allocations(op.output).shape
    val dataType = model.allocations(op.inputA).dataType

    if (shapeA == shapeB) {
      // ── Fast path ──────────────────────────────────────────────────────────
      val count = outputShape.product
      dataType match {
        case DataType.Float32 =>
          val a = memory(op.inputA).asInstanceOf[Ptr[CFloat]]
          val b = memory(op.inputB).asInstanceOf[Ptr[CFloat]]
          val o = memory(op.output).asInstanceOf[Ptr[CFloat]]
          var i = 0; while (i < count) { !(o + i) = !(a + i) * !(b + i); i += 1 }

        case DataType.Float64 =>
          val a = memory(op.inputA).asInstanceOf[Ptr[CDouble]]
          val b = memory(op.inputB).asInstanceOf[Ptr[CDouble]]
          val o = memory(op.output).asInstanceOf[Ptr[CDouble]]
          var i = 0; while (i < count) { !(o + i) = !(a + i) * !(b + i); i += 1 }

        case DataType.Int32 =>
          val a = memory(op.inputA).asInstanceOf[Ptr[CInt]]
          val b = memory(op.inputB).asInstanceOf[Ptr[CInt]]
          val o = memory(op.output).asInstanceOf[Ptr[CInt]]
          var i = 0; while (i < count) { !(o + i) = !(a + i) * !(b + i); i += 1 }

        case DataType.Int64 =>
          val a = memory(op.inputA).asInstanceOf[Ptr[CLongLong]]
          val b = memory(op.inputB).asInstanceOf[Ptr[CLongLong]]
          val o = memory(op.output).asInstanceOf[Ptr[CLongLong]]
          var i = 0; while (i < count) { !(o + i) = !(a + i) * !(b + i); i += 1 }

        case unsupported =>
          throw new NotImplementedError(s"Mul not implemented for data type: $unsupported")
      }
    } else {
      // ── Broadcast path ─────────────────────────────────────────────────────
      val outArr = outputShape.toArray
      val stA = computeBroadcastStrides(shapeA, outArr.length)
      val stB = computeBroadcastStrides(shapeB, outArr.length)

      var outFlat = 0
      dataType match {
        case DataType.Float32 =>
          val a = memory(op.inputA).asInstanceOf[Ptr[CFloat]]
          val b = memory(op.inputB).asInstanceOf[Ptr[CFloat]]
          val o = memory(op.output).asInstanceOf[Ptr[CFloat]]
          broadcastLoopN(outArr, Array(stA, stB)) { idx =>
            !(o + outFlat) = !(a + idx(0)) * !(b + idx(1))
            outFlat += 1
          }

        case DataType.Float64 =>
          val a = memory(op.inputA).asInstanceOf[Ptr[CDouble]]
          val b = memory(op.inputB).asInstanceOf[Ptr[CDouble]]
          val o = memory(op.output).asInstanceOf[Ptr[CDouble]]
          broadcastLoopN(outArr, Array(stA, stB)) { idx =>
            !(o + outFlat) = !(a + idx(0)) * !(b + idx(1))
            outFlat += 1
          }

        case DataType.Int32 =>
          val a = memory(op.inputA).asInstanceOf[Ptr[CInt]]
          val b = memory(op.inputB).asInstanceOf[Ptr[CInt]]
          val o = memory(op.output).asInstanceOf[Ptr[CInt]]
          broadcastLoopN(outArr, Array(stA, stB)) { idx =>
            !(o + outFlat) = !(a + idx(0)) * !(b + idx(1))
            outFlat += 1
          }

        case DataType.Int64 =>
          val a = memory(op.inputA).asInstanceOf[Ptr[CLongLong]]
          val b = memory(op.inputB).asInstanceOf[Ptr[CLongLong]]
          val o = memory(op.output).asInstanceOf[Ptr[CLongLong]]
          broadcastLoopN(outArr, Array(stA, stB)) { idx =>
            !(o + outFlat) = !(a + idx(0)) * !(b + idx(1))
            outFlat += 1
          }

        case unsupported =>
          throw new NotImplementedError(
            s"Broadcast Mul not implemented for data type: $unsupported",
          )
      }
    }
  }

  /** Handles casting between supported data types (Float32 <-> Float64). */
  private def handleCast(op: Operation.Cast, memory: MemoryMap, model: ModelIR): IO[Unit] = IO {
    val inputAlloc = model.allocations(op.input)
    val outputAlloc = model.allocations(op.output)
    val count = inputAlloc.shape.product
    val inputPtr = memory(op.input)
    val outputPtr = memory(op.output)

    (inputAlloc.dataType, outputAlloc.dataType) match {
      case (from, to) if from == to =>
        ()

      case (DataType.Float64, DataType.Float32) =>
        val in = inputPtr.asInstanceOf[Ptr[CDouble]]
        val out = outputPtr.asInstanceOf[Ptr[CFloat]]
        var i = 0
        while (i < count) {
          !(out + i) = (!(in + i)).toFloat
          i += 1
        }

      case (DataType.Float32, DataType.Float64) =>
        val in = inputPtr.asInstanceOf[Ptr[CFloat]]
        val out = outputPtr.asInstanceOf[Ptr[CDouble]]
        var i = 0
        while (i < count) {
          !(out + i) = (!(in + i)).toDouble
          i += 1
        }

      case (DataType.Int32, DataType.Bool) =>
        val in = inputPtr.asInstanceOf[Ptr[CInt]]
        val out = outputPtr.asInstanceOf[Ptr[Boolean]]
        var i = 0
        while (i < count) {
          !(out + i) = !(in + i) != 0
          i += 1
        }
      case (DataType.Int64, DataType.Bool) =>
        val in = inputPtr.asInstanceOf[Ptr[CLongLong]]
        val out = outputPtr.asInstanceOf[Ptr[Boolean]]
        var i = 0
        while (i < count) {
          !(out + i) = !(in + i) != 0
          i += 1
        }
      case (from, to) =>
        // Unreachable due to pre-validation.
        throw new IllegalStateException(s"Unvalidated cast from $from to $to encountered.")
    }
  }

  /** Handles the SVMClassifier operation by constructing a native LibSvm model, performing the
    * prediction, and writing the results to the output tensors.
    */
  /** Handles the SVMClassifier operation using the C++ wrapper for robust LibSVM integration. This
    * approach eliminates struct layout issues and provides ONNX-compliant per-class scores.
    */
  private def handleSvmClassifier(
      op: Operation.SVMClassifier,
      memory: MemoryMap,
      model: ModelIR,
  ): Resource[IO, IO[Unit]] = {
    val numFeatures = model.allocations(op.input).shape.last

    // Create SVM model using C++ wrapper with proper resource management
    createSvmModelResource(op, numFeatures).map { svmModel =>
      IO {
        // Get input and output pointers from memory map
        val inputPtr = memory(op.input).asInstanceOf[Ptr[CDouble]]
        val scoresPtr = memory(op.outputScores).asInstanceOf[Ptr[CDouble]]
        val labelPtr = memory(op.outputLabel).asInstanceOf[Ptr[CInt]]

        // Single function call - all complexity handled in C++
        val predictedLabel = svm_predict_with_scores(
          svmModel,
          inputPtr,
          numFeatures,
          scoresPtr,
        )

        // Write back the predicted label
        !labelPtr = predictedLabel

        ()
      }
    }
  }

  /** Creates an SVM model using the C++ wrapper functions with proper resource management. All
    * memory allocation and model construction is handled in C for maximum reliability.
    */
  private def createSvmModelResource(
      op: Operation.SVMClassifier,
      numFeatures: Int,
  ): Resource[IO, Ptr[Byte]] = {
    val nrClass = op.classLabels.size
    val numSupportVectors = op.vectorsPerClass.sum.toInt

    for {
      // Create SVM parameter using C++ wrapper
      param <- createSvmParameterResource(op)

      // Create managed arrays for model data
      supportVectorsPtr <- createManagedDoubleArray(op.supportVectors)
      coefficientsPtr <- createManagedDoubleArray(op.coefficients)
      rhoPtr <- createManagedDoubleArray(op.rho.toArray)
      classLabelsPtr <- createManagedIntArray(op.classLabels.map(_.toInt).toArray)
      vectorsPerClassPtr <- createManagedIntArray(op.vectorsPerClass.map(_.toInt).toArray)

      // Create the SVM model using C++ wrapper with LibSVM's native cleanup
      svmModel <- Resource.make(IO {
        create_svm_model(
          param,
          nrClass,
          numSupportVectors,
          supportVectorsPtr,
          numFeatures,
          coefficientsPtr,
          rhoPtr,
          classLabelsPtr,
          vectorsPerClassPtr,
        )
      })(model =>
        IO {
          // Use LibSVM's native cleanup function
          val modelPtrPtr = stdlib.malloc(sizeof[Ptr[Byte]]).asInstanceOf[Ptr[Ptr[Byte]]]
          !modelPtrPtr = model
          svm_free_and_destroy_model(modelPtrPtr)
          stdlib.free(modelPtrPtr.asInstanceOf[Ptr[Byte]])
        },
      )

    } yield svmModel
  }

  /** Creates SVM parameter using C++ wrapper with proper resource management. */
  private def createSvmParameterResource(op: Operation.SVMClassifier): Resource[IO, Ptr[Byte]] = {
    val kernelType = op.kernelType match {
      case SVMKernel.Linear => 0
      case SVMKernel.Poly => 1
      case SVMKernel.Rbf => 2
      case SVMKernel.Sigmoid => 3
    }

    val gamma = op.kernelParams.headOption.getOrElse(0.0)
    val coef0 = op.kernelParams.drop(1).headOption.getOrElse(0.0)
    val degree = op.kernelParams.drop(2).headOption.map(_.toInt).getOrElse(3)

    Resource.make(IO {
      create_svm_param(
        svm_type = 0, // C_SVC
        kernel_type = kernelType,
        degree = degree,
        gamma = gamma,
        coef0 = coef0,
      )
    })(param => IO(stdlib.free(param)))
  }

  /** Helper to create managed double array for C++ wrapper calls. */
  private def createManagedDoubleArray(values: Array[Double]): Resource[IO, Ptr[CDouble]] =
    Resource.make(IO {
      val ptr = stdlib
        .malloc(sizeof[CDouble] * values.length.toUSize)
        .asInstanceOf[Ptr[CDouble]]
      if (ptr == null) throw new OutOfMemoryError(s"Failed to allocate ${values.length} doubles")

      for (i <- values.indices)
        ptr(i) = values(i)
      ptr
    })(ptr => IO(stdlib.free(ptr.asInstanceOf[Ptr[Byte]])))

  /** Helper to create managed int array for C++ wrapper calls. */
  private def createManagedIntArray(values: Array[Int]): Resource[IO, Ptr[CInt]] =
    Resource.make(IO {
      val ptr = stdlib
        .malloc(sizeof[CInt] * values.length.toUSize)
        .asInstanceOf[Ptr[CInt]]
      if (ptr == null) throw new OutOfMemoryError(s"Failed to allocate ${values.length} ints")

      for (i <- values.indices)
        ptr(i) = values(i)
      ptr
    })(ptr => IO(stdlib.free(ptr.asInstanceOf[Ptr[Byte]])))

  /** Creates empty Scala arrays for each graph output. These arrays will be pointed to by the
    * `memoryResource` and written to directly from native code.
    */
  private def createOutputArrays(model: ModelIR): Map[String, Array[_]] =
    model.graphOutputs.map { name =>
      val allocation = model.allocations(name)
      val size = allocation.shape.product
      val array: Array[_] = allocation.dataType match {
        case DataType.Float32 => new Array[Float](size)
        case DataType.Int32 => new Array[Int](size)
        case DataType.Float64 => new Array[Double](size)
        case DataType.Int64 => new Array[Long](size)
        case DataType.Bool => new Array[Boolean](size)
        case other => throw new Exception(s"Unsupported output data type: $other")
      }
      name -> array
    }.toMap

  /** Handles ReLU activation function: output = max(0, input) Simple element-wise operation with
    * good performance in Scala.
    */
  private def handleRelu(op: Operation.Relu, memory: MemoryMap, model: ModelIR): IO[Unit] = IO {
    val count = model.allocations(op.output).shape.product
    val inputAlloc = model.allocations(op.input)

    inputAlloc.dataType match {
      case DataType.Float32 =>
        val input = memory(op.input).asInstanceOf[Ptr[CFloat]]
        val output = memory(op.output).asInstanceOf[Ptr[CFloat]]
        var i = 0
        while (i < count) {
          val value = !(input + i)
          !(output + i) = if (value > 0.0f) value else 0.0f
          i += 1
        }

      case DataType.Float64 =>
        val input = memory(op.input).asInstanceOf[Ptr[CDouble]]
        val output = memory(op.output).asInstanceOf[Ptr[CDouble]]
        var i = 0
        while (i < count) {
          val value = !(input + i)
          !(output + i) = if (value > 0.0) value else 0.0
          i += 1
        }

      case unsupported =>
        throw new NotImplementedError(
          s"ReLU operation not implemented for data type: $unsupported",
        ) // should never happen due to pre-validation
    }
  }

  /** Handles Reshape operation: changes the shape of a tensor without modifying its data. This is a
    * no-op in terms of data, but must ensure the output pointer is correctly set.
    */
  private def handleReshape(op: Operation.Reshape, memory: MemoryMap, model: ModelIR): IO[Unit] =
    IO {
      val inputAlloc = model.allocations(op.input)
      val inputPtr = memory(op.input)
      val outputPtr = memory(op.output)
      val totalBytes = (inputAlloc.shape.product * inputAlloc.dataType.sizeInBytes).toUSize

      // Copy data from input to output (same data, different shape interpretation)
      memcpy(outputPtr, inputPtr, totalBytes)
      ()
    }
  // BROADCASTING HELPER METHODS
  /** Calculates the broadcasted output shape following numpy broadcasting rules. Returns None if
    * shapes are incompatible for broadcasting.
    */
  private def calculateBroadcastShape(shapeA: List[Int], shapeB: List[Int]): Option[List[Int]] = {
    val maxDims = math.max(shapeA.length, shapeB.length)

    // Pad shapes with leading 1s
    val paddedA = List.fill(maxDims - shapeA.length)(1) ++ shapeA
    val paddedB = List.fill(maxDims - shapeB.length)(1) ++ shapeB

    // Use traverse to validate and transform dimensions
    paddedA.zip(paddedB).traverse { case (dimA, dimB) =>
      if (dimA == dimB) {
        Some(dimA)
      } else if (dimA == 1) {
        Some(dimB)
      } else if (dimB == 1) {
        Some(dimA)
      } else {
        None
      }
    }
  }

  private def handleConv(
      op: Operation.Conv,
      memory: MemoryMap,
      model: ModelIR,
  ): Resource[IO, IO[Unit]] = {

    // All your existing shape/pointer extraction logic
    val inputAlloc = model.allocations(op.input)
    val weightAlloc = model.allocations(op.weight)

    val inputShape = inputAlloc.shape
    val weightShape = weightAlloc.shape
    val (inputChannels, inputHeight, inputWidth) = (inputShape(1), inputShape(2), inputShape(3))
    val (outputChannels, kernelHeight, kernelWidth) =
      (weightShape(0), op.kernelShape(0), op.kernelShape(1))

    val inputPtr = memory(op.input)
    val weightPtr = memory(op.weight)
    val outputPtr = memory(op.output)
    val biasPtrOpt = op.bias.map(b => memory(b))

    val autoPadValue = op.autoPad match {
      case AutoPad.SameUpper => 1
      case AutoPad.Valid => 0
      case AutoPad.NotSet if op.pads.forall(_ == 0) => 0
      case other => throw new IllegalArgumentException(s"Unsupported autoPad in handleConv: $other")
    }

    val useBias = if (biasPtrOpt.isDefined) 1 else 0

    // Two-phase Resource pattern
    inputAlloc.dataType match {
      case DataType.Float32 =>
        Resource
          .make(IO {
            // Phase 1: Initialize with all your extracted values
            MLPack.initialise_conv_f(
              outputChannels.toUSize,
              kernelHeight.toUSize,
              kernelWidth.toUSize,
              op.strides(0).toUSize,
              op.strides(1).toUSize,
              autoPadValue,
              useBias,
              inputHeight.toUSize,
              inputWidth.toUSize,
              inputChannels.toUSize,
              weightPtr.asInstanceOf[Ptr[Float]],
              biasPtrOpt.getOrElse(null).asInstanceOf[Ptr[Float]],
              inputPtr.asInstanceOf[Ptr[Float]],
              outputPtr.asInstanceOf[Ptr[Float]],
            )
          })(handle =>
            IO {
              // Phase 3: Cleanup
              MLPack.cleanup_conv_f(handle)
            },
          )
          .map { handle =>
            // Phase 2: Execute
            IO(MLPack.execute_conv_f(handle))
          }

      case DataType.Float64 =>
        Resource
          .make(IO {
            MLPack.initialise_conv_d(
              outputChannels.toUSize,
              kernelHeight.toUSize,
              kernelWidth.toUSize,
              op.strides(0).toUSize,
              op.strides(1).toUSize,
              autoPadValue,
              useBias,
              inputHeight.toUSize,
              inputWidth.toUSize,
              inputChannels.toUSize,
              weightPtr.asInstanceOf[Ptr[Double]],
              biasPtrOpt.getOrElse(null).asInstanceOf[Ptr[Double]],
              inputPtr.asInstanceOf[Ptr[Double]],
              outputPtr.asInstanceOf[Ptr[Double]],
            )
          })(handle =>
            IO {
              MLPack.cleanup_conv_d(handle)
            },
          )
          .map { handle =>
            IO(MLPack.execute_conv_d(handle))
          }

      case other =>
        throw new NotImplementedError(s"Conv input data type $other not supported.")
    }
  }
  private def handleMaxPool(
      op: Operation.MaxPool,
      memory: MemoryMap,
      model: ModelIR,
  ): Resource[IO, IO[Unit]] = {

    // All your existing extraction logic
    val inputAlloc = model.allocations(op.input)
    val inputShape = inputAlloc.shape

    val (inputChannels, inputHeight, inputWidth) = (inputShape(1), inputShape(2), inputShape(3))
    val (kernelHeight, kernelWidth) = (op.kernelShape(0), op.kernelShape(1))

    val inputPtr = memory(op.input)
    val outputPtr = memory(op.output)

    // Two-phase Resource pattern
    inputAlloc.dataType match {
      case DataType.Float32 =>
        Resource
          .make(IO {
            // Phase 1: Initialize with all your extracted values
            MLPack.initialise_pool_f(
              kernelHeight.toUSize,
              kernelWidth.toUSize,
              op.strides(0).toUSize,
              op.strides(1).toUSize,
              inputHeight.toUSize,
              inputWidth.toUSize,
              inputChannels.toUSize,
              inputPtr.asInstanceOf[Ptr[Float]],
              outputPtr.asInstanceOf[Ptr[Float]],
            )
          })(handle =>
            IO {
              // Phase 3: Cleanup
              MLPack.cleanup_pool_f(handle)
            },
          )
          .map { handle =>
            // Phase 2: Execute
            IO(MLPack.execute_pool_f(handle))
          }

      case DataType.Float64 =>
        Resource
          .make(IO {
            MLPack.initialise_pool_d(
              kernelHeight.toUSize,
              kernelWidth.toUSize,
              op.strides(0).toUSize,
              op.strides(1).toUSize,
              inputHeight.toUSize,
              inputWidth.toUSize,
              inputChannels.toUSize,
              inputPtr.asInstanceOf[Ptr[Double]],
              outputPtr.asInstanceOf[Ptr[Double]],
            )
          })(handle =>
            IO {
              MLPack.cleanup_pool_d(handle)
            },
          )
          .map { handle =>
            IO(MLPack.execute_pool_d(handle))
          }

      case other =>
        throw new NotImplementedError(s"MaxPool input data type $other not supported.")
    }
  }

  /** Handles Matrix Multiplication using OpenBLAS CBLAS functions. Performs C = A * B where A is
    * [M, K], B is [K, N], and C is [M, N].
    */
  private def handleMatMul(op: Operation.MatMul, memory: MemoryMap, model: ModelIR): IO[Unit] = IO {
    val inputAAlloc = model.allocations(op.inputA)
    val inputBAlloc = model.allocations(op.inputB)

    // Re-apply promotion rules (cheap, fast, already validated)
    var shapeA = inputAAlloc.shape
    var shapeB = inputBAlloc.shape
    if (shapeA.length == 1) shapeA = 1 :: shapeA
    if (shapeB.length == 1) shapeB = shapeB :+ 1

    val M = shapeA(shapeA.length - 2)
    val K = shapeA(shapeA.length - 1)
    val N = shapeB(shapeB.length - 1)

    val batchA = shapeA.dropRight(2)
    val batchB = shapeB.dropRight(2)
    val batchRank = batchA.length.max(batchB.length)

    val paddedA = List.fill(batchRank - batchA.length)(1) ++ batchA
    val paddedB = List.fill(batchRank - batchB.length)(1) ++ batchB
    // Calculate final batch output shape natively
    val batchOut = paddedA.zip(paddedB).map { case (a, b) => a.max(b) }

    val stA = computeBroadcastStrides(paddedA ++ List(M, K), batchRank + 2).take(batchRank)
    val stB = computeBroadcastStrides(paddedB ++ List(K, N), batchRank + 2).take(batchRank)
    val stC = computeBroadcastStrides(batchOut ++ List(M, N), batchRank + 2).take(batchRank)

    // Unsafe pointer matching (safe because validateMatMul guaranteed the types)
    inputAAlloc.dataType match {
      case DataType.Float32 =>
        val a = memory(op.inputA).asInstanceOf[Ptr[CFloat]]
        val b = memory(op.inputB).asInstanceOf[Ptr[CFloat]]
        val c = memory(op.output).asInstanceOf[Ptr[CFloat]]

        broadcastLoopN(batchOut.toArray, Array(stA, stB, stC)) { idx =>
          cblas_sgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasNoTrans,
            M,
            N,
            K,
            1.0f,
            a + idx(0),
            K,
            b + idx(1),
            N,
            0.0f,
            c + idx(2),
            N,
          )
        }

      case DataType.Float64 =>
        val a = memory(op.inputA).asInstanceOf[Ptr[CDouble]]
        val b = memory(op.inputB).asInstanceOf[Ptr[CDouble]]
        val c = memory(op.output).asInstanceOf[Ptr[CDouble]]

        broadcastLoopN(batchOut.toArray, Array(stA, stB, stC)) { idx =>
          cblas_dgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasNoTrans,
            M,
            N,
            K,
            1.0,
            a + idx(0),
            K,
            b + idx(1),
            N,
            0.0,
            c + idx(2),
            N,
          )
        }

      case _ => () // Unreachable due to validation phase
    }
  }
  private def handleSoftmax(op: Operation.Softmax, memory: MemoryMap, model: ModelIR): IO[Unit] =
    IO {
      val inputAlloc = model.allocations(op.input)
      val shape = inputAlloc.shape
      val rank = shape.length

      // Normalize the ONNX axis (supports negative indexing)
      val axis = if (op.axis < 0) op.axis + rank else op.axis

      // Compute the matrix dimensions for the C++ reduction
      val outerSize = shape.take(axis).product
      val innerSize = shape.drop(axis).product

      val inputPtr = memory(op.input)
      val outputPtr = memory(op.output)

      inputAlloc.dataType match {
        case DataType.Float32 =>
          MLPack.F_perform_softmax_direct(
            inputPtr.asInstanceOf[Ptr[CFloat]],
            outerSize.toUSize,
            innerSize.toUSize,
            outputPtr.asInstanceOf[Ptr[CFloat]],
          )

        case DataType.Float64 =>
          MLPack.perform_softmax_direct(
            inputPtr.asInstanceOf[Ptr[CDouble]],
            outerSize.toUSize,
            innerSize.toUSize,
            outputPtr.asInstanceOf[Ptr[CDouble]],
          )

        case other =>
          // Unreachable due to validateModel, but required for exhaustiveness
          throw new NotImplementedError(s"Softmax input data type $other not supported.")
      }
    }

  private def handleGather(op: Operation.Gather, memory: MemoryMap, model: ModelIR): IO[Unit] = IO {
    val dataAlloc = model.allocations(op.input)
    val indicesAlloc = model.allocations(op.indices)
    val dataShape = dataAlloc.shape
    val elemSize = dataAlloc.dataType.sizeInBytes // bytes per scalar

    // Normalise axis (ONNX allows negative)
    val rank = dataShape.length
    val axis = if (op.axis < 0) op.axis + rank else op.axis

    val outerSize = dataShape.take(axis).product // product of dims before axis
    val axisSize = dataShape(axis) // extent we index into
    val innerSize = dataShape.drop(axis + 1).product // contiguous block after axis
    val numIndices = indicesAlloc.shape.product

    val dataPtr = memory(op.input)
    val idxPtr = memory(op.indices)
    val outputPtr = memory(op.output)

    var outer = 0
    while (outer < outerSize) {
      var i = 0
      while (i < numIndices) {

        // read the index
        val rawIdx = indicesAlloc.dataType match {
          case DataType.Int32 => (!(idxPtr.asInstanceOf[Ptr[CInt]] + i)).toLong
          case DataType.Int64 => !(idxPtr.asInstanceOf[Ptr[CLongLong]] + i)
          case other =>
            throw new IllegalStateException(
              s"Unexpected index dtype: $other",
            ) // should not reach due to early validation
        }
        // wrap around for -ve indices
        val idx = if (rawIdx < 0) rawIdx + axisSize else rawIdx
        require(
          idx >= 0 && idx < axisSize,
          s"Gather index $rawIdx out of bounds for axis size $axisSize",
        )

        val srcOffset = (outer * axisSize + idx) * innerSize
        val dstOffset = (outer * numIndices + i) * innerSize
        val byteCount = (innerSize * elemSize).toUSize

        memcpy(
          outputPtr + dstOffset * elemSize,
          dataPtr + srcOffset * elemSize,
          byteCount,
        )
        i += 1
      }
      outer += 1
    }
  }
  private def handleAnd(op: Operation.And, memory: MemoryMap, model: ModelIR) = IO {
    // first get shapes then execute fast path by taking data from memorymap
    val inputA = model.allocations(op.inputA)
    val inputB = model.allocations(op.inputB)
    val shapeA = inputA.shape
    val shapeB = inputB.shape

    val ptrA = memory(op.inputA).asInstanceOf[Ptr[Boolean]]
    val ptrB = memory(op.inputB).asInstanceOf[Ptr[Boolean]]
    val ptrOutput = memory(op.output).asInstanceOf[Ptr[Boolean]]
    if (shapeA == shapeB) {

      val outputCount = model.allocations(op.output).shape.product
      var i = 0
      while (i < outputCount) {
        !(ptrOutput + i) = !(ptrA + i) && !(ptrB + i)
        i += 1
      }
    } else {
      val outputShapeArray = model.allocations(op.output).shape.toArray
      var outIdx = 0
      val stridesA = computeBroadcastStrides(shapeA, outputShapeArray.length)
      val stridesB = computeBroadcastStrides(shapeB, outputShapeArray.length)
      broadcastLoopN(outputShapeArray, Array(stridesA, stridesB)) { idx =>
        !(ptrOutput + outIdx) = !(ptrA + idx(0)) && !(ptrB + idx(1))
        outIdx += 1
      }
    }

  }
  private def handleDiv(op: Operation.Div, memory: MemoryMap, model: ModelIR): IO[Unit] = IO {
    val shapeA = model.allocations(op.inputA).shape
    val shapeB = model.allocations(op.inputB).shape
    val outputShape = model.allocations(op.output).shape
    val dataType = model.allocations(op.inputA).dataType

    if (shapeA == shapeB) {
      val count = outputShape.product
      dataType match {
        case DataType.Float32 =>
          val a = memory(op.inputA).asInstanceOf[Ptr[CFloat]]
          val b = memory(op.inputB).asInstanceOf[Ptr[CFloat]]
          val o = memory(op.output).asInstanceOf[Ptr[CFloat]]
          var i = 0; while (i < count) { !(o + i) = !(a + i) / !(b + i); i += 1 }

        case DataType.Float64 =>
          val a = memory(op.inputA).asInstanceOf[Ptr[CDouble]]
          val b = memory(op.inputB).asInstanceOf[Ptr[CDouble]]
          val o = memory(op.output).asInstanceOf[Ptr[CDouble]]
          var i = 0; while (i < count) { !(o + i) = !(a + i) / !(b + i); i += 1 }

        case DataType.Int32 =>
          val a = memory(op.inputA).asInstanceOf[Ptr[CInt]]
          val b = memory(op.inputB).asInstanceOf[Ptr[CInt]]
          val o = memory(op.output).asInstanceOf[Ptr[CInt]]
          var i = 0; while (i < count) { !(o + i) = !(a + i) / !(b + i); i += 1 }

        case DataType.Int64 =>
          val a = memory(op.inputA).asInstanceOf[Ptr[CLongLong]]
          val b = memory(op.inputB).asInstanceOf[Ptr[CLongLong]]
          val o = memory(op.output).asInstanceOf[Ptr[CLongLong]]
          var i = 0; while (i < count) { !(o + i) = !(a + i) / !(b + i); i += 1 }

        case unsupported =>
          throw new NotImplementedError(s"Div not implemented for data type: $unsupported")
      }
    } else {
      val outArr = outputShape.toArray
      val stA = computeBroadcastStrides(shapeA, outArr.length)
      val stB = computeBroadcastStrides(shapeB, outArr.length)

      var outFlat = 0
      dataType match {
        case DataType.Float32 =>
          val a = memory(op.inputA).asInstanceOf[Ptr[CFloat]]
          val b = memory(op.inputB).asInstanceOf[Ptr[CFloat]]
          val o = memory(op.output).asInstanceOf[Ptr[CFloat]]
          broadcastLoopN(outArr, Array(stA, stB)) { idx =>
            !(o + outFlat) = !(a + idx(0)) / !(b + idx(1))
            outFlat += 1
          }

        case DataType.Float64 =>
          val a = memory(op.inputA).asInstanceOf[Ptr[CDouble]]
          val b = memory(op.inputB).asInstanceOf[Ptr[CDouble]]
          val o = memory(op.output).asInstanceOf[Ptr[CDouble]]
          broadcastLoopN(outArr, Array(stA, stB)) { idx =>
            !(o + outFlat) = !(a + idx(0)) / !(b + idx(1))
            outFlat += 1
          }

        case DataType.Int32 =>
          val a = memory(op.inputA).asInstanceOf[Ptr[CInt]]
          val b = memory(op.inputB).asInstanceOf[Ptr[CInt]]
          val o = memory(op.output).asInstanceOf[Ptr[CInt]]
          broadcastLoopN(outArr, Array(stA, stB)) { idx =>
            !(o + outFlat) = !(a + idx(0)) / !(b + idx(1))
            outFlat += 1
          }

        case DataType.Int64 =>
          val a = memory(op.inputA).asInstanceOf[Ptr[CLongLong]]
          val b = memory(op.inputB).asInstanceOf[Ptr[CLongLong]]
          val o = memory(op.output).asInstanceOf[Ptr[CLongLong]]
          broadcastLoopN(outArr, Array(stA, stB)) { idx =>
            !(o + outFlat) = !(a + idx(0)) / !(b + idx(1))
            outFlat += 1
          }

        case unsupported =>
          throw new NotImplementedError(
            s"Broadcast Div not implemented for data type: $unsupported",
          )
      }
    }
  }

  private def handleErf(op: Operation.Erf, memory: MemoryMap, model: ModelIR) = IO {
    val dataType = model.allocations(op.input).dataType
    val count = model.allocations(op.output).shape.product

    dataType match {
      case (DataType.Float32) =>
        val input = memory(op.input).asInstanceOf[Ptr[CFloat]]
        val output = memory(op.output).asInstanceOf[Ptr[CFloat]]
        var i = 0
        while (i < count) {
          !(output + i) = erff(!(input + i))
          i += 1
        }

      case (DataType.Float64) =>
        val input = memory(op.input).asInstanceOf[Ptr[CDouble]]
        val output = memory(op.output).asInstanceOf[Ptr[CDouble]]
        var i = 0
        while (i < count) {
          !(output + i) = erf(!(input + i))
          i += 1
        }

      case (_) =>
        throw new NotImplementedError("currently only Float32 and Float64 (Double) is handled")
    }

  }
  private def handleTanh(op: Operation.Tanh, memory: MemoryMap, model: ModelIR) = IO {
    val dataType = model.allocations(op.input).dataType
    val count = model.allocations(op.output).shape.product

    dataType match {
      case (DataType.Float32) =>
        val input = memory(op.input).asInstanceOf[Ptr[CFloat]]
        val output = memory(op.output).asInstanceOf[Ptr[CFloat]]
        var i = 0
        while (i < count) {
          !(output + i) = tanhf(!(input + i))
          i += 1
        }

      case (DataType.Float64) =>
        val input = memory(op.input).asInstanceOf[Ptr[CDouble]]
        val output = memory(op.output).asInstanceOf[Ptr[CDouble]]
        var i = 0
        while (i < count) {
          !(output + i) = tanh(!(input + i))
          i += 1
        }

      case (_) =>
        throw new NotImplementedError("currently only Float32 and Float64 (Double) is handled")
    }

  }
  private def handleGatherND(op: Operation.GatherND, memory: MemoryMap, model: ModelIR): IO[Unit] =
    IO {
      val dataAlloc = model.allocations(op.input)
      val indicesAlloc = model.allocations(op.indices)

      val dataShape = dataAlloc.shape
      val indicesShape = indicesAlloc.shape
      val elemSize = dataAlloc.dataType.sizeInBytes

      val batchDims = op.batchDims
      val indexDepth = indicesShape.last

      val outerSize = dataShape.take(batchDims).product
      val numIndices = indicesShape.drop(batchDims).dropRight(1).product
      val innerSize = dataShape.drop(batchDims + indexDepth).product

      val allDataStrides = calculateStrides(dataShape)

      val indexStrides = allDataStrides.slice(batchDims, batchDims + indexDepth)
      val indexedDims = dataShape.slice(batchDims, batchDims + indexDepth).toArray

      val dataBatchStride = dataShape.drop(batchDims).product
      val outputBatchStride = numIndices * innerSize

      val dataPtr = memory(op.input)
      val idxPtr = memory(op.indices)
      val outputPtr = memory(op.output)

      var outer = 0
      while (outer < outerSize) {
        var i = 0
        while (i < numIndices) {

          val indicesOffset = (outer * numIndices + i) * indexDepth

          var srcOffset = outer.toLong * dataBatchStride

          var k = 0
          while (k < indexDepth) {
            // read the index
            val rawIdx = indicesAlloc.dataType match {
              case DataType.Int32 => (!(idxPtr.asInstanceOf[Ptr[CInt]] + indicesOffset + k)).toLong
              case DataType.Int64 => !(idxPtr.asInstanceOf[Ptr[CLongLong]] + indicesOffset + k)
              case other =>
                throw new IllegalStateException(
                  s"Unexpected index dtype: $other",
                )
            }

            // wrap around for -ve indices
            val dimSize = indexedDims(k)
            val idx = if (rawIdx < 0) rawIdx + dimSize else rawIdx
            require(
              idx >= 0 && idx < dimSize,
              s"GatherND index $rawIdx out of bounds for axis size $dimSize",
            )

            srcOffset += (idx * indexStrides(k))
            k += 1
          }

          val dstOffset = outer.toLong * outputBatchStride + i.toLong * innerSize
          val byteCount = (innerSize * elemSize).toUSize

          memcpy(
            outputPtr + (dstOffset * elemSize),
            dataPtr + (srcOffset * elemSize),
            byteCount,
          )

          i += 1
        }
        outer += 1
      }
    }
  private def handleGemm(op: Operation.Gemm, memory: MemoryMap, model: ModelIR): IO[Unit] = IO {
    val inputAAlloc = model.allocations(op.inputA)
    val inputBAlloc = model.allocations(op.inputB)

    val shapeA = inputAAlloc.shape
    val shapeB = inputBAlloc.shape

    val transAFlag = if (op.transA != 0) CblasTrans else CblasNoTrans
    val transBFlag = if (op.transB != 0) CblasTrans else CblasNoTrans

    val M = if (op.transA != 0) shapeA(1) else shapeA(0)
    val K = if (op.transA != 0) shapeA(0) else shapeA(1)
    val N = if (op.transB != 0) shapeB(0) else shapeB(1)

    val lda = shapeA(1)
    val ldb = shapeB(1)
    val ldc = N

    val inputAPtr = memory(op.inputA)
    val inputBPtr = memory(op.inputB)
    val outputPtr = memory(op.output)

    // Handle Bias Broadcast
    val eps = 1e-6f
    val hasActiveBias = op.inputC.isDefined && math.abs(op.beta) > eps

    if (hasActiveBias) {
      val cName = op.inputC.get
      val cPtr = memory(cName)
      val cShape = model.allocations(cName).shape
      gemmBroadcastBias(M, N, cPtr, cShape, outputPtr, inputAAlloc.dataType)
    }

    // Execute BLAS
    val blasBeta = if (hasActiveBias) op.beta else 0.0f

    inputAAlloc.dataType match {
      case DataType.Float32 =>
        cblas_sgemm(
          layout = CblasRowMajor,
          transA = transAFlag,
          transB = transBFlag,
          M = M,
          N = N,
          K = K,
          alpha = op.alpha,
          A = inputAPtr.asInstanceOf[Ptr[CFloat]],
          lda = lda,
          B = inputBPtr.asInstanceOf[Ptr[CFloat]],
          ldb = ldb,
          beta = blasBeta,
          C = outputPtr.asInstanceOf[Ptr[CFloat]],
          ldc = ldc,
        )

      case DataType.Float64 =>
        cblas_dgemm(
          layout = CblasRowMajor,
          transA = transAFlag,
          transB = transBFlag,
          M = M,
          N = N,
          K = K,
          alpha = op.alpha.toDouble,
          A = inputAPtr.asInstanceOf[Ptr[CDouble]],
          lda = lda,
          B = inputBPtr.asInstanceOf[Ptr[CDouble]],
          ldb = ldb,
          beta = blasBeta.toDouble,
          C = outputPtr.asInstanceOf[Ptr[CDouble]],
          ldc = ldc,
        )

      case unsupported =>
        throw new IllegalStateException(s"Unvalidated Gemm data type: $unsupported")
    }
  }

  private def handleIsNaN(op: Operation.IsNaN, memory: MemoryMap, model: ModelIR) = IO {
    val inputShape = model.allocations(op.input).shape
    val count = inputShape.product
    val outPtr = memory(op.output).asInstanceOf[Ptr[Boolean]]
    val dataType = model.allocations(op.input).dataType

    dataType match {
      case DataType.Float32 =>
        val inpPtr = memory(op.input).asInstanceOf[Ptr[CFloat]]
        var i = 0
        while (i < count) {
          !(outPtr + i) = (!(inpPtr + i)).isNaN
          i += 1
        }
      case DataType.Float64 =>
        val inpPtr = memory(op.input).asInstanceOf[Ptr[CDouble]]
        var i = 0
        while (i < count) {
          !(outPtr + i) = (!(inpPtr + i)).isNaN
          i += 1
        }

      case _ =>
        throw new IllegalStateException(
          s"isNaN works only for float and double",
        ) // should not be reached due to early validation
    }
  }

  private def handleWhere(op: Operation.Where, memory: MemoryMap, model: ModelIR): IO[Unit] = IO {
    val shapeCond = model.allocations(op.condition).shape
    val shapeA = model.allocations(op.inputA).shape
    val shapeB = model.allocations(op.inputB).shape
    val outputShape = model.allocations(op.output).shape
    val dataType = model.allocations(op.inputA).dataType

    val condition = memory(op.condition).asInstanceOf[Ptr[Boolean]]
    val ptrA = memory(op.inputA)
    val ptrB = memory(op.inputB)
    val ptrOutput = memory(op.output)

    // Fast path: ALL THREE tensors must have the exact same shape
    if (shapeA == shapeB && shapeB == shapeCond) {
      val count = outputShape.product
      dataType match {
        case DataType.Float32 =>
          val a = ptrA.asInstanceOf[Ptr[CFloat]]
          val b = ptrB.asInstanceOf[Ptr[CFloat]]
          val o = ptrOutput.asInstanceOf[Ptr[CFloat]]
          var i = 0
          while (i < count) {
            !(o + i) = if (!(condition + i)) !(a + i) else !(b + i)
            i += 1
          }

        case DataType.Float64 =>
          val a = ptrA.asInstanceOf[Ptr[CDouble]]
          val b = ptrB.asInstanceOf[Ptr[CDouble]]
          val o = ptrOutput.asInstanceOf[Ptr[CDouble]]
          var i = 0
          while (i < count) {
            !(o + i) = if (!(condition + i)) !(a + i) else !(b + i)
            i += 1
          }

        case DataType.Int32 =>
          val a = ptrA.asInstanceOf[Ptr[CInt]]
          val b = ptrB.asInstanceOf[Ptr[CInt]]
          val o = ptrOutput.asInstanceOf[Ptr[CInt]]
          var i = 0
          while (i < count) {
            !(o + i) = if (!(condition + i)) !(a + i) else !(b + i)
            i += 1
          }

        case DataType.Int64 =>
          val a = ptrA.asInstanceOf[Ptr[CLongLong]]
          val b = ptrB.asInstanceOf[Ptr[CLongLong]]
          val o = ptrOutput.asInstanceOf[Ptr[CLongLong]]
          var i = 0
          while (i < count) {
            !(o + i) = if (!(condition + i)) !(a + i) else !(b + i)
            i += 1
          }

        case unsupported =>
          throw new NotImplementedError(
            s"Where operation not implemented for data type: $unsupported",
          )
      }
    } else {
      // Broadcasting path: Compute strides for all 3 inputs
      val outArr = outputShape.toArray
      val stCond = computeBroadcastStrides(shapeCond, outArr.length)
      val stA = computeBroadcastStrides(shapeA, outArr.length)
      val stB = computeBroadcastStrides(shapeB, outArr.length)

      var outFlat = 0
      dataType match {
        case DataType.Float32 =>
          val a = ptrA.asInstanceOf[Ptr[CFloat]]
          val b = ptrB.asInstanceOf[Ptr[CFloat]]
          val o = ptrOutput.asInstanceOf[Ptr[CFloat]]
          broadcastLoopN(outArr, Array(stCond, stA, stB)) { idx =>
            !(o + outFlat) = if (!(condition + idx(0))) !(a + idx(1)) else !(b + idx(2))
            outFlat += 1
          }

        case DataType.Float64 =>
          val a = ptrA.asInstanceOf[Ptr[CDouble]]
          val b = ptrB.asInstanceOf[Ptr[CDouble]]
          val o = ptrOutput.asInstanceOf[Ptr[CDouble]]
          broadcastLoopN(outArr, Array(stCond, stA, stB)) { idx =>
            !(o + outFlat) = if (!(condition + idx(0))) !(a + idx(1)) else !(b + idx(2))
            outFlat += 1
          }

        case DataType.Int32 =>
          val a = ptrA.asInstanceOf[Ptr[CInt]]
          val b = ptrB.asInstanceOf[Ptr[CInt]]
          val o = ptrOutput.asInstanceOf[Ptr[CInt]]
          broadcastLoopN(outArr, Array(stCond, stA, stB)) { idx =>
            !(o + outFlat) = if (!(condition + idx(0))) !(a + idx(1)) else !(b + idx(2))
            outFlat += 1
          }

        case DataType.Int64 =>
          val a = ptrA.asInstanceOf[Ptr[CLongLong]]
          val b = ptrB.asInstanceOf[Ptr[CLongLong]]
          val o = ptrOutput.asInstanceOf[Ptr[CLongLong]]
          broadcastLoopN(outArr, Array(stCond, stA, stB)) { idx =>
            !(o + outFlat) = if (!(condition + idx(0))) !(a + idx(1)) else !(b + idx(2))
            outFlat += 1
          }

        case unsupported =>
          throw new NotImplementedError(
            s"Broadcast Where not implemented for data type: $unsupported",
          )
      }
    }
  }

  /** pre-fill the output buffer before BLAS */
  private def gemmBroadcastBias(
      M: Int,
      N: Int,
      cPtr: Ptr[Byte],
      cShape: List[Int],
      yPtr: Ptr[Byte],
      dataType: DataType,
  ): Unit = {
    val totalElements = M.toLong * N
    val elemSize = dataType.sizeInBytes

    dataType match {
      case DataType.Float32 =>
        val c = cPtr.asInstanceOf[Ptr[CFloat]]
        val y = yPtr.asInstanceOf[Ptr[CFloat]]

        if (cShape.product == 1) {
          // Scalar: (), (1,), or (1,1)
          val scalarVal = !c
          var i = 0L; while (i < totalElements) { !(y + i) = scalarVal; i += 1L }
        } else if (cShape.length == 1 || cShape(0) == 1) {
          // Row-wise broadcast: (N,) or (1, N)
          var row = 0L
          val bytesPerRow = (N.toLong * elemSize).toUSize
          while (row < M) {
            memcpy(yPtr + (row * N * elemSize), cPtr, bytesPerRow)
            row += 1L
          }
        } else if (cShape.length == 2 && cShape(1) == 1) {
          // Column-wise broadcast: (M, 1)
          var row = 0L
          while (row < M) {
            val rowOffset = row * N
            val colVal = !(c + row)
            var col = 0L
            while (col < N) { !(y + rowOffset + col) = colVal; col += 1L }
            row += 1L
          }
        } else {
          // Full Matrix: (M, N)
          memcpy(yPtr, cPtr, (totalElements * elemSize).toUSize)
          ()
        }

      case DataType.Float64 =>
        val c = cPtr.asInstanceOf[Ptr[CDouble]]
        val y = yPtr.asInstanceOf[Ptr[CDouble]]

        if (cShape.product == 1) {
          val scalarVal = !c
          var i = 0L; while (i < totalElements) { !(y + i) = scalarVal; i += 1L }
        } else if (cShape.length == 1 || cShape(0) == 1) {
          var row = 0L
          val bytesPerRow = (N.toLong * elemSize).toUSize
          while (row < M) {
            memcpy(yPtr + (row * N * elemSize), cPtr, bytesPerRow)
            row += 1L
          }
        } else if (cShape.length == 2 && cShape(1) == 1) {
          var row = 0L
          while (row < M) {
            val rowOffset = row * N
            val colVal = !(c + row)
            var col = 0L
            while (col < N) { !(y + rowOffset + col) = colVal; col += 1L }
            row += 1L
          }
        } else {
          memcpy(yPtr, cPtr, (totalElements * elemSize).toUSize)
          ()
        }

      case _ => throw new IllegalStateException("Unsupported dataType for gemmBroadcastBias")
    }
  }
  private def handleTranspose(
      op: Operation.Transpose,
      memory: MemoryMap,
      model: ModelIR,
  ): IO[Unit] = IO {
    val inputAlloc = model.allocations(op.input)
    val outputAlloc = model.allocations(op.output)
    val rank = inputAlloc.shape.length

    val actualPerm = if (op.perm.isEmpty) (rank - 1 to 0 by -1).toList else op.perm

    val inStrides = calculateStrides(inputAlloc.shape)

    val permutedInStrides = actualPerm.map(i => inStrides(i)).toArray
    val outShape = outputAlloc.shape

    val inPtr = memory(op.input)
    val outPtr = memory(op.output)

    inputAlloc.dataType match {
      case DataType.Float32 =>
        val in = inPtr.asInstanceOf[Ptr[CFloat]]
        val out = outPtr.asInstanceOf[Ptr[CFloat]]
        transposeLoop(outShape, permutedInStrides) { (inFlat, outFlat) =>
          !(out + outFlat) = !(in + inFlat)

        }

      case DataType.Float64 =>
        val in = inPtr.asInstanceOf[Ptr[CDouble]]
        val out = outPtr.asInstanceOf[Ptr[CDouble]]
        transposeLoop(outShape, permutedInStrides) { (inFlat, outFlat) =>
          !(out + outFlat) = !(in + inFlat)

        }

      case DataType.Int32 =>
        val in = inPtr.asInstanceOf[Ptr[CInt]]
        val out = outPtr.asInstanceOf[Ptr[CInt]]
        transposeLoop(outShape, permutedInStrides) { (inFlat, outFlat) =>
          !(out + outFlat) = !(in + inFlat)

        }

      case DataType.Int64 =>
        val in = inPtr.asInstanceOf[Ptr[CLongLong]]
        val out = outPtr.asInstanceOf[Ptr[CLongLong]]
        transposeLoop(outShape, permutedInStrides) { (inFlat, outFlat) =>
          !(out + outFlat) = !(in + inFlat)

        }

      case unsupported =>
        throw new NotImplementedError(s"Transpose not implemented for data type: $unsupported")
    }
  }
  private def handleLayerNormalization(
      op: Operation.LayerNormalization,
      memory: MemoryMap,
      model: ModelIR,
  ): IO[Unit] = IO {
    val inputAlloc = model.allocations(op.input)
    val inputShape = inputAlloc.shape
    val rank = inputShape.length

    // Normalise axis — ONNX allows negative
    val axis = if (op.axis < 0) op.axis + rank else op.axis

    // outerSize = number of independent vectors to normalise
    // normSize  = length of each vector (the dims we reduce over)
    val outerSize = inputShape.take(axis).product
    val normSize = inputShape.drop(axis).product

    inputAlloc.dataType match {

      case DataType.Float32 =>
        val in = memory(op.input).asInstanceOf[Ptr[CFloat]]
        val scale = memory(op.scale).asInstanceOf[Ptr[CFloat]]
        val bias = op.bias.map(b => memory(b).asInstanceOf[Ptr[CFloat]])
        val out = memory(op.output).asInstanceOf[Ptr[CFloat]]
        val meanOut = op.mean.map(m => memory(m).asInstanceOf[Ptr[CFloat]])
        val invStdOut = op.inverseStdDeviation.map(s => memory(s).asInstanceOf[Ptr[CFloat]])
        val eps = op.epsilon

        var outer = 0
        while (outer < outerSize) {
          val base = outer * normSize

          //  Compute mean
          var sum = 0.0f
          var j = 0
          while (j < normSize) { sum += !(in + base + j); j += 1 }
          val mean = sum / normSize

          //  Compute variance
          var varAcc = 0.0f
          j = 0
          while (j < normSize) {
            val diff = !(in + base + j) - mean
            varAcc += diff * diff
            j += 1
          }
          val variance = varAcc / normSize

          //  Compute inverse std dev: 1 / sqrt(variance + epsilon)
          val invStd = 1.0f / sqrtf(variance + eps)

          //  Write optional training outputs
          meanOut.foreach(p => !(p + outer) = mean)
          invStdOut.foreach(p => !(p + outer) = invStd)

          //  Normalise, scale, and shift
          j = 0
          while (j < normSize) {
            val normalised = (!(in + base + j) - mean) * invStd
            val scaled = normalised * !(scale + j)
            !(out + base + j) = bias.fold(scaled)(b => scaled + !(b + j))
            j += 1
          }

          outer += 1
        }

      case DataType.Float64 =>
        val in = memory(op.input).asInstanceOf[Ptr[CDouble]]
        val scale = memory(op.scale).asInstanceOf[Ptr[CDouble]]
        val bias = op.bias.map(b => memory(b).asInstanceOf[Ptr[CDouble]])
        val out = memory(op.output).asInstanceOf[Ptr[CDouble]]
        val meanOut = op.mean.map(m => memory(m).asInstanceOf[Ptr[CDouble]])
        val invStdOut = op.inverseStdDeviation.map(s => memory(s).asInstanceOf[Ptr[CDouble]])
        val eps = op.epsilon.toDouble

        var outer = 0
        while (outer < outerSize) {
          val base = outer * normSize

          var sum = 0.0
          var j = 0
          while (j < normSize) { sum += !(in + base + j); j += 1 }
          val mean = sum / normSize

          var varAcc = 0.0
          j = 0
          while (j < normSize) {
            val diff = !(in + base + j) - mean
            varAcc += diff * diff
            j += 1
          }
          val variance = varAcc / normSize
          val invStd = 1.0 / sqrt(variance + eps)

          meanOut.foreach(p => !(p + outer) = mean)
          invStdOut.foreach(p => !(p + outer) = invStd)

          j = 0
          while (j < normSize) {
            val normalised = (!(in + base + j) - mean) * invStd
            val scaled = normalised * !(scale + j)
            !(out + base + j) = bias.fold(scaled)(b => scaled + !(b + j))
            j += 1
          }

          outer += 1
        }

      case unsupported =>
        throw new IllegalStateException(s"Unvalidated LayerNorm dtype: $unsupported")
    }
  }
  private def transposeLoop(
      outShape: List[Int],
      permutedInStrides: Array[Int],
  )(operation: (Int, Int) => Unit): Unit = {
    val rank = outShape.length
    val outputCount = outShape.product

    val coords = new Array[Int](rank)
    var inFlat = 0
    var outFlat = 0

    while (outFlat < outputCount) {
      operation(inFlat, outFlat)

      var dim = rank - 1
      while (dim >= 0) {
        coords(dim) += 1
        inFlat += permutedInStrides(dim)

        if (coords(dim) < outShape(dim)) {
          dim = -1
        } else {
          coords(dim) = 0
          inFlat -= permutedInStrides(dim) * outShape(dim)
          dim -= 1
        }
      }
      outFlat += 1
    }
  }

  /** Calculate row-major strides for a given shape */
  private def calculateStrides(shape: List[Int]): Array[Int] = {
    val strides = Array.fill(shape.length)(1)
    for (i <- shape.length - 2 to 0 by -1)
      strides(i) = strides(i + 1) * shape(i + 1)
    strides
  }

  /** Returns Broadcast strides calculated against an output rank, stride = 0 for dim 1 */
  private def computeBroadcastStrides(inputShape: List[Int], outputRank: Int): Array[Int] = {
    val strides = Array.fill(outputRank)(0)
    val offset = outputRank - inputShape.length
    val rawStrides = calculateStrides(inputShape)

    for (i <- inputShape.indices)
      strides(offset + i) = if (inputShape(i) == 1) 0 else rawStrides(i)

    strides
  }
  private def broadcastLoopN(
      outputShape: Array[Int],
      inputStrides: Array[Array[Int]],
  )(operation: Array[Int] => Unit): Unit = {
    val rank = outputShape.length
    val outputCount = {
      var p = 1; var i = 0
      while (i < rank) { p *= outputShape(i); i += 1 }
      p
    }
    val numInputs = inputStrides.length
    val coords = new Array[Int](rank)
    val inputIndices = new Array[Int](numInputs)

    var flat = 0
    while (flat < outputCount) {
      operation(inputIndices)

      var dim = rank - 1
      while (dim >= 0) {
        coords(dim) += 1

        var t = 0
        while (t < numInputs) {
          inputIndices(t) += inputStrides(t)(dim)
          t += 1
        }

        if (coords(dim) < outputShape(dim)) {
          dim = -1
        } else {

          coords(dim) = 0
          val extent = outputShape(dim)
          var t2 = 0
          while (t2 < numInputs) {
            inputIndices(t2) -= inputStrides(t2)(dim) * extent
            t2 += 1
          }
          dim -= 1
        }
      }
      flat += 1
    }
  }
}
