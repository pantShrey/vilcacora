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

import cats.effect.IO
import java.io.InputStream
import vilcacora.onnx.proto.ModelProto

object ModelLoader {

  /** Loads an ONNX model from classpath as a ModelProto. Use a forward slash for resource path,
    * e.g., "/mnist.onnx".
    */
  def loadModelFromPath(modelPath: String): IO[ModelProto] =
    IO.blocking {
      val stream: InputStream = getClass.getResourceAsStream(modelPath)
      if (stream == null)
        throw new IllegalArgumentException(s"Resource not found: $modelPath")
      try ModelProto.parseFrom(stream)

      finally stream.close()
    }
}
