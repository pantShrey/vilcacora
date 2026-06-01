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

package vilcacora.onnx

import com.armanbilge.vilcacora.ir.{DataType, Operation, PostTransform, SVMKernel, AutoPad}
import vilcacora.onnx.proto.{AttributeProto, ModelProto, NodeProto, TensorProto}
import com.google.protobuf.ByteString
import munit.FunSuite
import java.io.InputStream
import java.nio.{ByteBuffer, ByteOrder}

class TranslatorSuite extends FunSuite {

  // --- Unit tests for individual node translations ---

  test("translateNode should translate a MatMul node") {
    val node = NodeProto(opType = "MatMul", input = Seq("A", "B"), output = Seq("C"))
    assertEquals(Translator.translateNode(node), Right(Operation.MatMul("A", "B", "C")))
  }

  test("translateNode should translate an Add node") {
    val node = NodeProto(opType = "Add", input = Seq("X", "Y"), output = Seq("Z"))
    assertEquals(Translator.translateNode(node), Right(Operation.Add("X", "Y", "Z")))
  }

  test("translateNode should translate a Mul node") {
    val node = NodeProto(opType = "Mul", input = Seq("A", "B"), output = Seq("C"))
    assertEquals(Translator.translateNode(node), Right(Operation.Mul("A", "B", "C")))
  }

  test("translateNode should translate a Cast node") {
    val node = NodeProto(
      opType = "Cast",
      input = Seq("in"),
      output = Seq("out"),
      attribute = Seq(
        AttributeProto(name = "to", i = 7, `type` = AttributeProto.AttributeType.INT),
      ), // 7 = INT64
    )
    assertEquals(Translator.translateNode(node), Right(Operation.Cast("in", "out", DataType.Int64)))
  }

  // --- REWRITTEN TEST FOR SVMCLASSIFIER ---
  test(
    "translateNode should correctly translate an SVMClassifier node with floating point tolerance",
  ) {
    val node = NodeProto(
      opType = "SVMClassifier",
      input = Seq("features"),
      output = Seq("label", "scores"),
      attribute = Seq(
        AttributeProto(name = "classlabels_ints", ints = Seq(0L, 1L)),
        AttributeProto(name = "coefficients", floats = Seq(1.5f, -1.5f)),
        AttributeProto(name = "kernel_params", floats = Seq(0.5f, 0.1f)), // Note: 0.1f is inexact
        AttributeProto(name = "kernel_type", s = ByteString.copyFromUtf8("RBF")),
        AttributeProto(name = "post_transform", s = ByteString.copyFromUtf8("SOFTMAX")),
        AttributeProto(name = "rho", floats = Seq(0.9f)),
        AttributeProto(name = "support_vectors", floats = Seq(1.1f, 2.2f)),
        AttributeProto(name = "vectors_per_class", ints = Seq(1L, 1L)),
      ),
    )

    val result = Translator.translateNode(node)
    assert(result.isRight, "SVMClassifier translation failed unexpectedly")

    // Define a small tolerance for comparing floating point numbers
    val epsilon = 1e-6

    result.foreach {
      case op: Operation.SVMClassifier =>
        assertEquals(op.classLabels, List(0L, 1L))
        assertEquals(op.kernelType, SVMKernel.Rbf)
        assertEquals(op.postTransform, PostTransform.Softmax)

        // Use assertEquals with a tolerance (epsilon) for doubles
        assertEqualsDouble(op.kernelParams.head, 0.5, epsilon)
        assertEqualsDouble(op.kernelParams(1), 0.1, epsilon)
        assertEqualsDouble(op.rho.head, 0.9, epsilon)

        // Compare arrays of doubles element-by-element with tolerance
        op.coefficients.zip(Array(1.5, -1.5)).foreach { case (actual, expected) =>
          assertEqualsDouble(actual, expected, epsilon)
        }
        op.supportVectors.zip(Array(1.1, 2.2)).foreach { case (actual, expected) =>
          assertEqualsDouble(actual, expected, epsilon)
        }
      case other => fail(s"Expected SVMClassifier but got $other")
    }
  }

  test("translateNode should translate a Div node") {
    val node =
      NodeProto(opType = "Div", input = Seq("Numerator", "Denominator"), output = Seq("Quotient"))
    assertEquals(
      Translator.translateNode(node),
      Right(Operation.Div("Numerator", "Denominator", "Quotient")),
    )
  }

  test("translateNode should translate a Relu node") {
    val node = NodeProto(opType = "Relu", input = Seq("X"), output = Seq("Y"))
    assertEquals(Translator.translateNode(node), Right(Operation.Relu("X", "Y")))
  }

  test("translateNode should translate a Reshape node") {
    val node = NodeProto(opType = "Reshape", input = Seq("data", "shape"), output = Seq("reshaped"))
    // Test with the default 'allowzero' attribute
    assertEquals(
      Translator.translateNode(node),
      Right(Operation.Reshape("data", "shape", "reshaped", allowzero = false)),
    )
  }

  test("translateNode should translate a Constant node") {
    // Prepare the raw byte data for a Float32 tensor with value [1.0f, 2.0f]
    val byteBuffer = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN)
    byteBuffer.putFloat(1.0f)
    byteBuffer.putFloat(2.0f)
    val rawBytes = byteBuffer.array()

    // The constant data is stored as a TensorProto inside an AttributeProto
    val tensorProto = TensorProto(
      dataType = 1, // Float32
      dims = Seq(2),
      rawData = ByteString.copyFrom(rawBytes),
    )
    val attribute = AttributeProto(
      name = "value",
      t = Some(tensorProto),
      `type` = AttributeProto.AttributeType.TENSOR,
    )

    val node = NodeProto(
      opType = "Constant",
      input = Nil,
      output = Seq("const_out"),
      attribute = Seq(attribute),
    )

    val result = Translator.translateNode(node)
    assert(result.isRight, "Constant translation failed")
    result.foreach {
      case op: Operation.Constant =>
        assertEquals(op.output, "const_out")
        assertEquals(op.dataType, DataType.Float32)
        assertEquals(op.shape, List(2))
        assert(op.value.sameElements(rawBytes), "Raw byte data did not match")
      case other => fail(s"Expected Constant but got $other")
    }
  }

  test("translateNode should translate a Conv node") {
    val node = NodeProto(
      opType = "Conv",
      input = Seq("X", "W", "B"), // Input, Weights, Bias
      output = Seq("Y"),
      attribute = Seq(
        AttributeProto(name = "kernel_shape", ints = Seq(3L, 3L)),
        AttributeProto(name = "strides", ints = Seq(1L, 1L)),
        AttributeProto(name = "pads", ints = Seq(1L, 1L, 1L, 1L)),
        AttributeProto(name = "dilations", ints = Seq(1L, 1L)),
        AttributeProto(name = "group", i = 1L),
        AttributeProto(name = "auto_pad", s = ByteString.copyFromUtf8("NOTSET")),
      ),
    )

    val expected = Operation.Conv(
      input = "X",
      weight = "W",
      bias = Some("B"),
      output = "Y",
      autoPad = AutoPad.NotSet,
      dilations = List(1, 1),
      group = 1,
      kernelShape = List(3, 3),
      pads = List(1, 1, 1, 1),
      strides = List(1, 1),
    )

    assertEquals(Translator.translateNode(node), Right(expected))
  }

  test("translateNode should translate a Conv node without bias") {
    val node = NodeProto(
      opType = "Conv",
      input = Seq("X", "W"), // No bias
      output = Seq("Y"),
      attribute = Seq(
        AttributeProto(name = "kernel_shape", ints = Seq(1L, 1L)),
      ),
    )

    val result = Translator.translateNode(node)
    assert(result.isRight, "Conv without bias failed")
    result.foreach {
      case op: Operation.Conv => assertEquals(op.bias, None)
      case other => fail(s"Expected Conv but got $other")
    }
  }

  test("translateNode should translate a MaxPool node") {
    val node = NodeProto(
      opType = "MaxPool",
      input = Seq("X"),
      output = Seq("Y"),
      attribute = Seq(
        AttributeProto(name = "kernel_shape", ints = Seq(2L, 2L)),
        AttributeProto(name = "strides", ints = Seq(2L, 2L)),
        AttributeProto(name = "pads", ints = Seq(0L, 0L, 0L, 0L)),
        AttributeProto(name = "ceil_mode", i = 0L),
        AttributeProto(name = "storage_order", i = 0L),
        AttributeProto(name = "dilations", ints = Seq(1L, 1L)),
        AttributeProto(name = "auto_pad", s = ByteString.copyFromUtf8("NOTSET")),
      ),
    )

    val expected = Operation.MaxPool(
      input = "X",
      output = "Y",
      autoPad = AutoPad.NotSet,
      ceilMode = false,
      dilations = List(1, 1),
      kernelShape = List(2, 2),
      pads = List(0, 0, 0, 0),
      storageOrder = 0,
      strides = List(2, 2),
    )

    assertEquals(Translator.translateNode(node), Right(expected))
  }

  test("translateNode should translate a Softmax node with default axis") {
    val node = NodeProto(
      opType = "Softmax",
      input = Seq("input_tensor"),
      output = Seq("output_tensor"),
      // No axis attribute, should default to -1
    )

    assertEquals(
      Translator.translateNode(node),
      Right(Operation.Softmax("input_tensor", "output_tensor", axis = -1)),
    )
  }

  test("translateNode should translate a Softmax node with explicit axis") {
    val node = NodeProto(
      opType = "Softmax",
      input = Seq("logits"),
      output = Seq("probabilities"),
      attribute = Seq(
        AttributeProto(name = "axis", i = 1L, `type` = AttributeProto.AttributeType.INT),
      ),
    )

    assertEquals(
      Translator.translateNode(node),
      Right(Operation.Softmax("logits", "probabilities", axis = 1)),
    )
  }

  test("translateNode should handle Softmax with axis=0") {
    val node = NodeProto(
      opType = "Softmax",
      input = Seq("batch_logits"),
      output = Seq("batch_probs"),
      attribute = Seq(
        AttributeProto(name = "axis", i = 0L, `type` = AttributeProto.AttributeType.INT),
      ),
    )

    assertEquals(
      Translator.translateNode(node),
      Right(Operation.Softmax("batch_logits", "batch_probs", axis = 0)),
    )
  }

  test("translateNode should handle Gather with axis") {
    val node = NodeProto(
      opType = "Gather",
      input = List("input_tensor", "indices"),
      output = List("output_tensor"),
      attribute = List(
        AttributeProto(name = "axis", i = 1L, `type` = AttributeProto.AttributeType.INT),
      ),
    )

    assertEquals(
      Translator.translateNode(node),
      Right(
        Operation.Gather(
          input = "input_tensor",
          indices = "indices",
          output = "output_tensor",
          axis = 1,
        ),
      ),
    )
  }

  test("translateNode should handle Gather with default axis") {
    val node = NodeProto(
      opType = "Gather",
      input = List("input_tensor", "indices"),
      output = List("output_tensor"),
    )

    assertEquals(
      Translator.translateNode(node),
      Right(
        Operation.Gather(
          input = "input_tensor",
          indices = "indices",
          output = "output_tensor",
          axis = 0,
        ),
      ),
    )
  }

  test("translateNode should translate an And node") {
    val node = NodeProto(opType = "And", input = Seq("A", "B"), output = Seq("C"))
    assertEquals(Translator.translateNode(node), Right(Operation.And("A", "B", "C")))
  }

  test("translateNode should translate a BiasGelu node") {
    val node = NodeProto(opType = "BiasGelu", input = Seq("X", "bias"), output = Seq("Y"))
    assertEquals(Translator.translateNode(node), Right(Operation.BiasGelu("X", "bias", "Y")))
  }

  test("translateNode should translate a Clip node with all inputs") {
    val node = NodeProto(
      opType = "Clip",
      input = Seq("input", "min_val", "max_val"),
      output = Seq("clipped"),
    )
    assertEquals(
      Translator.translateNode(node),
      Right(Operation.Clip("input", Some("min_val"), Some("max_val"), "clipped")),
    )
  }

  test("translateNode should translate a Clip node with only min") {
    val node = NodeProto(
      opType = "Clip",
      input = Seq("input", "min_val"),
      output = Seq("clipped"),
    )
    assertEquals(
      Translator.translateNode(node),
      Right(Operation.Clip("input", Some("min_val"), None, "clipped")),
    )
  }

  test("translateNode should translate a Clip node with no min or max") {
    val node = NodeProto(opType = "Clip", input = Seq("input"), output = Seq("clipped"))
    assertEquals(
      Translator.translateNode(node),
      Right(Operation.Clip("input", None, None, "clipped")),
    )
  }

  test("translateNode should translate a Concat node with multiple inputs") {
    val node = NodeProto(
      opType = "Concat",
      input = Seq("T1", "T2", "T3"),
      output = Seq("out"),
      attribute = Seq(
        AttributeProto(name = "axis", i = 1L, `type` = AttributeProto.AttributeType.INT),
      ),
    )
    assertEquals(
      Translator.translateNode(node),
      Right(Operation.Concat(List("T1", "T2", "T3"), "out", axis = 1)),
    )
  }

  test("translateNode should translate a Concat node with negative axis") {
    val node = NodeProto(
      opType = "Concat",
      input = Seq("A", "B"),
      output = Seq("out"),
      attribute = Seq(
        AttributeProto(name = "axis", i = -1L, `type` = AttributeProto.AttributeType.INT),
      ),
    )
    assertEquals(
      Translator.translateNode(node),
      Right(Operation.Concat(List("A", "B"), "out", axis = -1)),
    )
  }

  test("translateNode should translate an Expand node") {
    val node = NodeProto(opType = "Expand", input = Seq("input", "shape"), output = Seq("output"))
    assertEquals(
      Translator.translateNode(node),
      Right(Operation.Expand("input", "shape", "output")),
    )
  }

  test("translateNode should translate a GatherElements node with default axis") {
    val node = NodeProto(
      opType = "GatherElements",
      input = Seq("data", "indices"),
      output = Seq("output"),
    )
    assertEquals(
      Translator.translateNode(node),
      Right(Operation.GatherElements("data", "indices", "output", axis = 0)),
    )
  }

  test("translateNode should translate a GatherElements node with explicit axis") {
    val node = NodeProto(
      opType = "GatherElements",
      input = Seq("data", "indices"),
      output = Seq("output"),
      attribute = Seq(
        AttributeProto(name = "axis", i = 2L, `type` = AttributeProto.AttributeType.INT),
      ),
    )
    assertEquals(
      Translator.translateNode(node),
      Right(Operation.GatherElements("data", "indices", "output", axis = 2)),
    )
  }

  test("translateNode should translate a GatherND node with default batchDims") {
    val node = NodeProto(
      opType = "GatherND",
      input = Seq("data", "indices"),
      output = Seq("output"),
    )
    assertEquals(
      Translator.translateNode(node),
      Right(Operation.GatherND("data", "indices", "output", batchDims = 0)),
    )
  }

  test("translateNode should translate a GatherND node with explicit batchDims") {
    val node = NodeProto(
      opType = "GatherND",
      input = Seq("data", "indices"),
      output = Seq("output"),
      attribute = Seq(
        AttributeProto(name = "batch_dims", i = 1L, `type` = AttributeProto.AttributeType.INT),
      ),
    )
    assertEquals(
      Translator.translateNode(node),
      Right(Operation.GatherND("data", "indices", "output", batchDims = 1)),
    )
  }

  test("translateNode should translate a GreaterOrEqual node") {
    val node = NodeProto(opType = "GreaterOrEqual", input = Seq("A", "B"), output = Seq("C"))
    assertEquals(
      Translator.translateNode(node),
      Right(Operation.GreaterOrEqual("A", "B", "C")),
    )
  }

  test("translateNode should translate a LayerNormalization node with all inputs and outputs") {
    val node = NodeProto(
      opType = "LayerNormalization",
      input = Seq("input", "scale", "bias"),
      output = Seq("Y", "mean", "inv_std_dev"),
      attribute = Seq(
        AttributeProto(name = "axis", i = -1L, `type` = AttributeProto.AttributeType.INT),
        AttributeProto(name = "epsilon", f = 1e-5f, `type` = AttributeProto.AttributeType.FLOAT),
        AttributeProto(name = "stash_type", i = 1L, `type` = AttributeProto.AttributeType.INT),
      ),
    )
    assertEquals(
      Translator.translateNode(node),
      Right(
        Operation.LayerNormalization(
          input = "input",
          scale = "scale",
          bias = Some("bias"),
          output = "Y",
          mean = Some("mean"),
          inverseStdDeviation = Some("inv_std_dev"),
          axis = -1,
          epsilon = 1e-5f,
          stashType = 1,
        ),
      ),
    )
  }

  test("translateNode should translate a LayerNormalization node with required inputs only") {
    // Only input + scale, no bias; only Y output, no mean or invStdDev
    val node = NodeProto(
      opType = "LayerNormalization",
      input = Seq("input", "scale"),
      output = Seq("Y"),
    )
    assertEquals(
      Translator.translateNode(node),
      Right(
        Operation.LayerNormalization(
          input = "input",
          scale = "scale",
          bias = None,
          output = "Y",
          mean = None,
          inverseStdDeviation = None,
        ),
      ),
    )
  }

  test("translateNode should translate a Max node with multiple inputs") {
    val node = NodeProto(
      opType = "Max",
      input = Seq("A", "B", "C"),
      output = Seq("max_out"),
    )
    assertEquals(
      Translator.translateNode(node),
      Right(Operation.Max(List("A", "B", "C"), "max_out")),
    )
  }

  test("translateNode should translate a Max node with a single input") {
    val node = NodeProto(opType = "Max", input = Seq("A"), output = Seq("out"))
    assertEquals(Translator.translateNode(node), Right(Operation.Max(List("A"), "out")))
  }

  test("translateNode should translate a Range node") {
    val node = NodeProto(
      opType = "Range",
      input = Seq("start", "limit", "delta"),
      output = Seq("output"),
    )
    assertEquals(
      Translator.translateNode(node),
      Right(Operation.Range("start", "limit", "delta", "output")),
    )
  }

  test("translateNode should translate a ReduceL2 node with axes input") {
    val node = NodeProto(
      opType = "ReduceL2",
      input = Seq("data", "axes"),
      output = Seq("reduced"),
      attribute = Seq(
        AttributeProto(name = "keepdims", i = 1L, `type` = AttributeProto.AttributeType.INT),
      ),
    )
    assertEquals(
      Translator.translateNode(node),
      Right(
        Operation.ReduceL2(
          input = "data",
          axes = Some("axes"),
          output = "reduced",
          keepDims = 1,
          noopWithEmptyAxes = 0,
        ),
      ),
    )
  }

  test("translateNode should translate a ReduceL2 node without axes and keepDims=0") {
    val node = NodeProto(
      opType = "ReduceL2",
      input = Seq("data"),
      output = Seq("reduced"),
      attribute = Seq(
        AttributeProto(name = "keepdims", i = 0L, `type` = AttributeProto.AttributeType.INT),
      ),
    )
    assertEquals(
      Translator.translateNode(node),
      Right(Operation.ReduceL2("data", None, "reduced", keepDims = 0, noopWithEmptyAxes = 0)),
    )
  }

  test("translateNode should translate a ReduceSum node with axes input") {
    val node = NodeProto(
      opType = "ReduceSum",
      input = Seq("data", "axes"),
      output = Seq("sum_out"),
      attribute = Seq(
        AttributeProto(name = "keepdims", i = 1L, `type` = AttributeProto.AttributeType.INT),
        AttributeProto(
          name = "noop_with_empty_axes",
          i = 1L,
          `type` = AttributeProto.AttributeType.INT,
        ),
      ),
    )
    assertEquals(
      Translator.translateNode(node),
      Right(
        Operation.ReduceSum(
          input = "data",
          axes = Some("axes"),
          output = "sum_out",
          keepDims = 1,
          noopWithEmptyAxes = 1,
        ),
      ),
    )
  }

  test("translateNode should translate a ReduceSum node without axes") {
    val node = NodeProto(opType = "ReduceSum", input = Seq("data"), output = Seq("out"))
    assertEquals(
      Translator.translateNode(node),
      Right(Operation.ReduceSum("data", None, "out", keepDims = 1, noopWithEmptyAxes = 0)),
    )
  }

  test("translateNode should translate a Shape node with default start") {
    val node = NodeProto(opType = "Shape", input = Seq("data"), output = Seq("shape_out"))
    assertEquals(
      Translator.translateNode(node),
      Right(Operation.Shape("data", "shape_out", end = None, start = 0)),
    )
  }

  test("translateNode should translate a Shape node with start and end attributes") {
    val node = NodeProto(
      opType = "Shape",
      input = Seq("data"),
      output = Seq("shape_out"),
      attribute = Seq(
        AttributeProto(name = "start", i = 1L, `type` = AttributeProto.AttributeType.INT),
        AttributeProto(name = "end", i = 3L, `type` = AttributeProto.AttributeType.INT),
      ),
    )
    assertEquals(
      Translator.translateNode(node),
      Right(Operation.Shape("data", "shape_out", end = Some(3), start = 1)),
    )
  }

  test("translateNode should translate a Slice node with required inputs only") {
    val node = NodeProto(
      opType = "Slice",
      input = Seq("data", "starts", "ends"),
      output = Seq("sliced"),
    )
    assertEquals(
      Translator.translateNode(node),
      Right(
        Operation.Slice(
          input = "data",
          starts = "starts",
          ends = "ends",
          axes = None,
          steps = None,
          output = "sliced",
        ),
      ),
    )
  }

  test("translateNode should translate a Slice node with axes and steps") {
    val node = NodeProto(
      opType = "Slice",
      input = Seq("data", "starts", "ends", "axes", "steps"),
      output = Seq("sliced"),
    )
    assertEquals(
      Translator.translateNode(node),
      Right(
        Operation.Slice(
          input = "data",
          starts = "starts",
          ends = "ends",
          axes = Some("axes"),
          steps = Some("steps"),
          output = "sliced",
        ),
      ),
    )
  }

  test("translateNode should translate a Slice node with axes but no steps") {
    val node = NodeProto(
      opType = "Slice",
      input = Seq("data", "starts", "ends", "axes"),
      output = Seq("sliced"),
    )
    assertEquals(
      Translator.translateNode(node),
      Right(
        Operation.Slice(
          input = "data",
          starts = "starts",
          ends = "ends",
          axes = Some("axes"),
          steps = None,
          output = "sliced",
        ),
      ),
    )
  }

  test("translateNode should translate a Squeeze node with axes input") {
    val node = NodeProto(
      opType = "Squeeze",
      input = Seq("data", "axes"),
      output = Seq("squeezed"),
    )
    assertEquals(
      Translator.translateNode(node),
      Right(Operation.Squeeze("data", Some("axes"), "squeezed")),
    )
  }

  test("translateNode should translate a Squeeze node without axes") {
    val node = NodeProto(opType = "Squeeze", input = Seq("data"), output = Seq("squeezed"))
    assertEquals(
      Translator.translateNode(node),
      Right(Operation.Squeeze("data", None, "squeezed")),
    )
  }

  test("translateNode should translate a Transpose node with explicit perm") {
    val node = NodeProto(
      opType = "Transpose",
      input = Seq("X"),
      output = Seq("Y"),
      attribute = Seq(
        AttributeProto(
          name = "perm",
          ints = Seq(0L, 2L, 1L),
          `type` = AttributeProto.AttributeType.INTS,
        ),
      ),
    )
    assertEquals(
      Translator.translateNode(node),
      Right(Operation.Transpose("X", "Y", perm = List(0, 2, 1))),
    )
  }

  test("translateNode should translate a Transpose node with no perm (reverse all dims)") {
    // Absent perm means reverse all dimensions; we represent this as empty list
    val node = NodeProto(opType = "Transpose", input = Seq("X"), output = Seq("Y"))
    assertEquals(
      Translator.translateNode(node),
      Right(Operation.Transpose("X", "Y", perm = List.empty)),
    )
  }

  test("translateNode should translate an Unsqueeze node") {
    val node = NodeProto(
      opType = "Unsqueeze",
      input = Seq("data", "axes"),
      output = Seq("expanded"),
    )
    assertEquals(
      Translator.translateNode(node),
      Right(Operation.Unsqueeze("data", "axes", "expanded")),
    )
  }

  test("translateNode should translate a Where node") {
    val node = NodeProto(
      opType = "Where",
      input = Seq("condition", "X", "Y"),
      output = Seq("output"),
    )
    assertEquals(
      Translator.translateNode(node),
      Right(Operation.Where("condition", "X", "Y", "output")),
    )
  }

  // --- Arity / error handling tests ---

  test("translateNode should fail for a MatMul node with wrong arity") {
    val node = NodeProto(opType = "MatMul", input = Seq("A"), output = Seq("C"))
    assert(Translator.translateNode(node).isLeft, "Expected Left for wrong-arity MatMul")
  }

  test("translateNode should fail for a Range node with too few inputs") {
    val node = NodeProto(opType = "Range", input = Seq("start", "limit"), output = Seq("output"))
    assert(Translator.translateNode(node).isLeft, "Expected Left for Range with only 2 inputs")
  }

  test("translateNode should fail for a Slice node with too few inputs") {
    val node = NodeProto(opType = "Slice", input = Seq("data", "starts"), output = Seq("out"))
    assert(Translator.translateNode(node).isLeft, "Expected Left for Slice missing ends")
  }

  test("translateNode should fail for a Concat node with no inputs") {
    val node = NodeProto(
      opType = "Concat",
      input = Seq.empty,
      output = Seq("out"),
      attribute = Seq(AttributeProto(name = "axis", i = 0L)),
    )
    assert(Translator.translateNode(node).isLeft, "Expected Left for Concat with no inputs")
  }
  test("translateNode should translate an Erf node") {
    val node = NodeProto(opType = "Erf", input = Seq("X"), output = Seq("Y"))
    assertEquals(Translator.translateNode(node), Right(Operation.Erf("X", "Y")))
  }

  test("translateNode should translate an IsNaN node") {
    val node = NodeProto(opType = "IsNaN", input = Seq("X"), output = Seq("Y"))
    assertEquals(Translator.translateNode(node), Right(Operation.IsNaN("X", "Y")))
  }

  test("translateNode should translate a Tanh node") {
    val node = NodeProto(opType = "Tanh", input = Seq("X"), output = Seq("Y"))
    assertEquals(Translator.translateNode(node), Right(Operation.Tanh("X", "Y")))
  }

  test("translateNode should translate a Gemm node with all inputs and explicit attributes") {
    val node = NodeProto(
      opType = "Gemm",
      input = Seq("A", "B", "C"),
      output = Seq("Y"),
      attribute = Seq(
        AttributeProto(name = "alpha", f = 0.5f, `type` = AttributeProto.AttributeType.FLOAT),
        AttributeProto(name = "beta", f = 2.0f, `type` = AttributeProto.AttributeType.FLOAT),
        AttributeProto(name = "transA", i = 1L, `type` = AttributeProto.AttributeType.INT),
        AttributeProto(name = "transB", i = 0L, `type` = AttributeProto.AttributeType.INT),
      ),
    )
    assertEquals(
      Translator.translateNode(node),
      Right(
        Operation.Gemm("A", "B", Some("C"), "Y", alpha = 0.5f, beta = 2.0f, transA = 1, transB = 0),
      ),
    )
  }

  test("translateNode should translate a Gemm node without bias and default attributes") {
    val node = NodeProto(opType = "Gemm", input = Seq("A", "B"), output = Seq("Y"))
    assertEquals(
      Translator.translateNode(node),
      Right(Operation.Gemm("A", "B", None, "Y", alpha = 1f, beta = 1f, transA = 0, transB = 0)),
    )
  }

  test("translateNode should fail for a Gemm node with wrong arity") {
    val node = NodeProto(opType = "Gemm", input = Seq("A"), output = Seq("Y"))
    assert(Translator.translateNode(node).isLeft, "Expected Left for Gemm with only 1 input")
  }
  test("translateNode should return Left for an unsupported operation type") {
    val node = NodeProto(opType = "UnknownOp", input = Seq("X"), output = Seq("Y"))
    assert(Translator.translateNode(node).isLeft, "Expected Left for unsupported op")
    Translator.translateNode(node).left.foreach { err =>
      assert(err.contains("Unsupported operation type"), s"Unexpected error message: $err")
    }
  }

  // --- End-to-end tests using model files from resources ---

  // Helper now includes the leading slash, which is the correct way to specify an absolute path on the classpath
  def loadModelFromPath(modelPath: String): ModelProto = {
    val stream: InputStream = getClass.getResourceAsStream(modelPath)
    if (stream == null)
      throw new IllegalArgumentException(s"Resource not found: $modelPath")
    try ModelProto.parseFrom(stream)
    finally stream.close()
  }

  test("translate should successfully process a valid, fully static model") {
    // This test now depends on the file 'static_svm.onnx' being in 'onnx/src/test/resources'
    val modelProto = loadModelFromPath("/static_svm.onnx")
    val irResult = Translator.translate(modelProto)

    assert(irResult.isRight, "Translation failed unexpectedly for a valid model")
    irResult.foreach { modelIR =>
      assertEquals(modelIR.graphInputs, List("features_float32x30"))
      assertEquals(modelIR.operations.length, 5)
    }
  }

  test("translate should fail gracefully for a model with dynamic shapes") {
    // This test now depends on the file 'dynamic_svm.onnx' being in 'onnx/src/test/resources'
    val modelProto = loadModelFromPath("/dynamic_svm.onnx")
    val irResult = Translator.translate(modelProto)

    assert(irResult.isLeft, "Translation should have failed but it succeeded")
    irResult.left.foreach { error =>
      assert(error.contains("unknown dimension"))
    }
  }
}
