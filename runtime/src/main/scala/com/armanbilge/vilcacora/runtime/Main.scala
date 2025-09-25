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

import cats.effect.{IO, IOApp}
import com.armanbilge.vilcacora.ir._
import vilcacora.onnx.Translator
import vilcacora.onnx.proto.ModelProto
import java.nio.file.{Files, Paths}

object MainApp extends IOApp.Simple {

  // Helper to load the ONNX model from a file in the project's root directory.
  private def loadModelFromFile(path: String): IO[ModelProto] = IO {
    println(s"Loading model from: $path")
    val bytes = Files.readAllBytes(Paths.get(path))
    ModelProto.parseFrom(bytes)
  }

  // Create a simple synthetic MNIST-like input (28x28 grayscale image)
  // This creates a simple pattern that resembles a digit for testing
  private def createMNISTInput(): Array[Float] = {
    val image = Array.ofDim[Float](1 * 1 * 28 * 28) // Shape: [1, 1, 28, 28]

    // Create a simple "7"-like pattern in the center
    // MNIST expects white foreground (1.0) on black background (0.0)
    for {
      h <- 8 until 20 // height range
      w <- 6 until 22 // width range
    } {
      val idx = h * 28 + w
      if (h == 8 || (w >= 18 && h <= 15)) {
        image(idx) = 1.0f // White pixels for the "7" shape
      }
    }

    image
  }

  private def printImagePreview(image: Array[Float], width: Int = 28, height: Int = 28): Unit = {
    println("Input image preview (28x28):")
    for (h <- 0 until height) {
      for (w <- 0 until width) {
        val pixel = image(h * width + w)
        if (pixel > 0.5f) print("█") else print("·")
      }
      println()
    }
    println()
  }

  private def printClassificationResults(probabilities: Array[Float]): Unit = {
    println("Classification results:")
    println("Digit | Probability")
    println("------|------------")

    probabilities.zipWithIndex.foreach { case (prob, digit) =>
      println(f"  $digit   | ${prob * 100}%6.2f%%")
    }

    val predictedClass = probabilities.zipWithIndex.maxBy(_._1)._2
    val confidence = probabilities.max * 100
    println()
    println(f"Predicted digit: $predictedClass (confidence: $confidence%.2f%%)")
  }

  def run: IO[Unit] = {
    // 1. Load and translate model (unchanged)
    val modelIRIO: IO[ModelIR] = for {
      modelProto <- loadModelFromFile("mnist12_static.onnx")
      _ <- IO.println("MNIST-12 model loaded. Translating to ModelIR...")
      modelIR <- IO.fromEither(
        Translator.translate(modelProto).left.map { errorMsg =>
          new RuntimeException(s"Translation failed: $errorMsg")
        },
      )
      _ <- IO.println("Translation successful.")
      _ <- IO.println(s"Model has ${modelIR.operations.length} operations")
      _ <- IO.println(s"Graph inputs: ${modelIR.graphInputs.mkString(", ")}")
      _ <- IO.println(s"Graph outputs: ${modelIR.graphOutputs.mkString(", ")}")
    } yield modelIR

    modelIRIO
      .flatMap { modelIR =>
        // 2. Create input data
        val inputTensorName = modelIR.graphInputs.headOption.getOrElse("Input3")
        val inputShape =
          modelIR.allocations.get(inputTensorName).map(_.shape).getOrElse(List(1, 1, 28, 28))

        IO.println(s"Input tensor: '$inputTensorName' with shape: ${inputShape.mkString("x")}") >>
          IO {
            val inputImage = createMNISTInput()
            printImagePreview(inputImage)
            inputImage
          }.flatMap { inputImage =>
            val inputData: Map[String, Array[Float]] = Map(inputTensorName -> inputImage)

            // 3. **SINGLE .use() CALL** - Initialize once, execute multiple times
            Interpreter.execute(modelIR, inputData).use { runMainModel =>
              for {
                // First execution: Main MNIST model
                _ <- IO.println("Executing main MNIST model...")
                mainResults <- runMainModel
                _ <- IO.println("Main model inference successful! Raw logits received.")

                // Print raw output
                _ <- IO {
                  println("\n--- Raw Model Output (Logits) ---")
                  mainResults.foreach { case (name, array) =>
                    println(s"Output tensor '$name':")
                    array match {
                      case arr: Array[Float] => println(s"  ${arr.mkString("Array(", ", ", ")")}")
                      case arr: Array[Double] => println(s"  ${arr.mkString("Array(", ", ", ")")}")
                      case arr => println(s"  Unexpected raw output type: $arr")
                    }
                  }
                  println("------------------------------------")
                }

                // 4. Create softmax model and execute it in a SEPARATE resource scope
                (outputTensorName, rawOutputArray) = mainResults.head
                originalOutputAllocation = modelIR.allocations(outputTensorName)

                softmaxInputName = "softmax_input"
                softmaxOutputName = "softmax_output"

                softmaxOp = Operation.Softmax(input = softmaxInputName, output = softmaxOutputName)

                softmaxModel = ModelIR(
                  name = "runtime-softmax",
                  operations = List(softmaxOp),
                  allocations = Map(
                    softmaxInputName -> Allocation(
                      softmaxInputName,
                      originalOutputAllocation.dataType,
                      originalOutputAllocation.shape,
                    ),
                    softmaxOutputName -> Allocation(
                      softmaxOutputName,
                      originalOutputAllocation.dataType,
                      originalOutputAllocation.shape,
                    ),
                  ),
                  graphInputs = List(softmaxInputName),
                  graphOutputs = List(softmaxOutputName),
                )

                softmaxInputData = Map(softmaxInputName -> rawOutputArray)

                // 5. **SEPARATE .use() CALL for Softmax** - This is OK since it's a different model
                _ <- IO.println("\nApplying Softmax operation to the output...")
                finalResults <- Interpreter.execute(softmaxModel, softmaxInputData).use {
                  runSoftmax =>
                    runSoftmax.flatMap { softmaxOutputMap =>
                      IO.println("Softmax applied successfully!") >>
                        IO {
                          println("\n--- MNIST Classification Results (after Softmax) ---")
                          val probabilitiesArray = softmaxOutputMap(softmaxOutputName)

                          probabilitiesArray match {
                            case probArray: Array[Float] if probArray.length == 10 =>
                              printClassificationResults(probArray)
                            case probArray: Array[Double] if probArray.length == 10 =>
                              val floatArray = probArray.map(_.toFloat)
                              printClassificationResults(floatArray)
                            case arr: Array[_] =>
                              println(
                                s"Unexpected softmax output: ${arr.mkString("Array(", ", ", ")")}",
                              )
                          }
                        }
                    }
                }
              } yield finalResults
            }
          }
      }
      .handleErrorWith { err =>
        IO.println(s"An error occurred: ${err.getMessage}") >>
          IO.println(s"Stack trace: ${err.getStackTrace.take(10).mkString("\n")}")
      }
  }

}
