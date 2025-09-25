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

import scala.scalanative.unsafe._

/** Scala Native bindings for MLPack C++ wrapper functions Two-phase operations: initialization and
  * execution
  */
@linkCppRuntime
@extern
object MLPack {

  /* opaque handle types */
  type ConvHandleF = Ptr[Byte]
  type ConvHandleD = Ptr[Byte]
  type PoolHandleF = Ptr[Byte]
  type PoolHandleD = Ptr[Byte]
  type SoftmaxF = Ptr[Byte]
  type SoftmaxD = Ptr[Byte]

  /* ---- convolution ---- */
  def initialise_conv_f(
      outMaps: CSize,
      kH: CSize,
      kW: CSize,
      sH: CSize,
      sW: CSize,
      autoPad: CInt,
      useBias: CInt,
      inH: CSize,
      inW: CSize,
      inC: CSize,
      weight: Ptr[Float],
      bias: Ptr[Float],
      inputPtr: Ptr[Float],
      outputPtr: Ptr[Float],
  ): ConvHandleF = extern

  def execute_conv_f(h: ConvHandleF): Unit = extern
  def cleanup_conv_f(h: ConvHandleF): Unit = extern

  def initialise_conv_d(
      outMaps: CSize,
      kH: CSize,
      kW: CSize,
      sH: CSize,
      sW: CSize,
      autoPad: CInt,
      useBias: CInt,
      inH: CSize,
      inW: CSize,
      inC: CSize,
      weight: Ptr[Double],
      bias: Ptr[Double],
      inputPtr: Ptr[Double],
      outputPtr: Ptr[Double],
  ): ConvHandleD = extern

  def execute_conv_d(h: ConvHandleD): Unit = extern
  def cleanup_conv_d(h: ConvHandleD): Unit = extern

  /* ---- max-pool ---- */
  def initialise_pool_f(
      kH: CSize,
      kW: CSize,
      sH: CSize,
      sW: CSize,
      inH: CSize,
      inW: CSize,
      inC: CSize,
      inputPtr: Ptr[Float],
      outputPtr: Ptr[Float],
  ): PoolHandleF = extern
  def execute_pool_f(h: PoolHandleF): Unit = extern
  def cleanup_pool_f(h: PoolHandleF): Unit = extern

  def initialise_pool_d(
      kH: CSize,
      kW: CSize,
      sH: CSize,
      sW: CSize,
      inH: CSize,
      inW: CSize,
      inC: CSize,
      inputPtr: Ptr[Double],
      outputPtr: Ptr[Double],
  ): PoolHandleD = extern
  def execute_pool_d(h: PoolHandleD): Unit = extern
  def cleanup_pool_d(h: PoolHandleD): Unit = extern

  /* ---- softmax ---- */
  def F_perform_softmax_direct(
      input_ptr: Ptr[CFloat],
      input_size: CSize,
      output_ptr: Ptr[CFloat], // Same size as input
  ): Unit = extern
  def perform_softmax_direct(
      input_ptr: Ptr[Double],
      input_size: CSize,
      output_ptr: Ptr[Double], // Same size as input
  ): Unit = extern
}
