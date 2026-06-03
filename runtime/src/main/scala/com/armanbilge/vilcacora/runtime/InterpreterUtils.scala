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
import com.armanbilge.vilcacora.ir._
import org.typelevel.keypool.KeyPool
import scala.concurrent.duration._

/** Utility functions for managing inference sessions and input buffers */
object InterpreterUtils {

  /** Safely copy data from one array to another using System.arraycopy wrapped in IO */
  def copyArrayToBuffer[A](
      source: Array[A],
      sourcePos: Int,
      dest: Array[A],
      destPos: Int,
      length: Int,
  ): IO[Unit] =
    IO {
      if (source == null || dest == null)
        throw new NullPointerException("Source and destination arrays must not be null")
      if (sourcePos < 0 || destPos < 0 || length < 0)
        throw new IllegalArgumentException("Positions and length must be non-negative")
      if (sourcePos + length > source.length)
        throw new IndexOutOfBoundsException("Source range out of bounds")
      if (destPos + length > dest.length)
        throw new IndexOutOfBoundsException("Destination range out of bounds")
      System.arraycopy(source, sourcePos, dest, destPos, length)
    }

  /** Creates a zero-filled buffer matching given shape and data type */
  def createBuffer(shape: List[Int], dataType: DataType): Array[_] = {
    val size = shape.product
    dataType match {
      case DataType.Float32 => new Array[Float](size)
      case DataType.Float64 => new Array[Double](size)
      case DataType.Int32 => new Array[Int](size)
      case DataType.Int64 => new Array[Long](size)
      case DataType.Bool => new Array[Boolean](size)
      case other =>
        throw new IllegalArgumentException(s"Unsupported data type: $other")
    }
  }

  /** Creates input buffers for all graph inputs of a model */
  def createInputBuffers(modelIR: ModelIR): Map[String, Array[_]] =
    modelIR.graphInputs.map { name =>
      val alloc = modelIR.allocations(name)
      name -> createBuffer(alloc.shape, alloc.dataType)
    }.toMap

  /** Wrapper representing an inference session */
  final case class InferenceSession(
      inputs: Map[String, Array[_]],
      runInference: IO[Map[String, Array[_]]],
  )

  /** Internal wrapper used by the pool to manage resource lifecycle. Only `.session` should be used
    * from this — the `release` is handled automatically by the pool.
    */
  final case class ManagedInferenceSession(
      session: InferenceSession,
      release: IO[Unit],
  )

  /** A resource managing a single inference session lifecycle */
  def inferenceSessionResource(
      modelIR: ModelIR,
      inputs: Map[String, Array[_]],
  ): Resource[IO, InferenceSession] =
    for {
      run <- Interpreter.execute(modelIR, inputs)
    } yield InferenceSession(inputs, run)

  /** Creates a KeyPool for managing concurrent inference sessions.
    *
    * @param modelIR
    *   The translated model IR
    * @param inputFactory
    *   Optional factory for creating input buffers for each inference session. If not provided,
    *   input buffers will be automatically allocated based on the model's graph input definitions.
    * @param maxTotal
    *   Maximum total concurrent sessions
    * @param maxPerKey
    *   Maximum sessions per key (default is Int.MaxValue)
    * @param idleTimeout
    *   Duration after which idle sessions are removed (default infinite)
    * @return
    *   Resource managing the inference session KeyPool
    */
  def inferenceSessionPool(
      modelIR: ModelIR,
      inputFactory: Option[() => Map[String, Array[_]]] = None,
      maxTotal: Int = 4,
      maxPerKey: Int = Int.MaxValue,
      idleTimeout: Duration = Duration.Inf,
  ): Resource[IO, KeyPool[IO, Unit, ManagedInferenceSession]] = {
    val factory = inputFactory.getOrElse(() => createInputBuffers(modelIR))
    KeyPool
      .Builder[IO, Unit, ManagedInferenceSession](
        create = (_: Unit) => {
          val inputs = factory()
          inferenceSessionResource(modelIR, inputs).allocated
            .map { case (session, release) =>
              ManagedInferenceSession(session, release)
            }
        },
        destroy = (managed: ManagedInferenceSession) => managed.release,
      )
      .withMaxPerKey(_ => maxPerKey)
      .withMaxTotal(maxTotal)
      .withIdleTimeAllowedInPool(idleTimeout)
      .build
  }
}
