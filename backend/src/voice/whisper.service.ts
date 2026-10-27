import { spawn } from "node:child_process";
import { promises as fs } from "node:fs";
import os from "node:os";
import path from "node:path";

export type TranscriptionResult = {
  text: string;
};

const WHISPER_CLI_PATH =
  process.env.WHISPER_CLI_PATH ?? "/opt/homebrew/bin/whisper-cli";

const WHISPER_MODEL_PATH =
  process.env.WHISPER_MODEL_PATH ??
  "/Users/mdshahariarhossen/prozyn-ai/models/ggml-base.bin";

export class WhisperService {
  async transcribe(audioFilePath: string): Promise<TranscriptionResult> {
    const outputDirectory = await fs.mkdtemp(
      path.join(os.tmpdir(), "prozyn-whisper-"),
    );

    try {
      const outputPrefix = path.join(outputDirectory, "transcription");

      const text = await this.runWhisper(
        audioFilePath,
        outputPrefix,
      );

      return {
        text: text.trim(),
      };
    } finally {
      await fs.rm(outputDirectory, {
        recursive: true,
        force: true,
      });
    }
  }

  private runWhisper(
    audioFilePath: string,
    outputPrefix: string,
  ): Promise<string> {
    return new Promise((resolve, reject) => {
      const args = [
        "-m",
        WHISPER_MODEL_PATH,
        "-f",
        audioFilePath,
        "-l",
        "en",
        "-nt",
        "-otxt",
        "-of",
        outputPrefix,
      ];

      const process = spawn(WHISPER_CLI_PATH, args);

      let stderr = "";

      process.stderr.on("data", (chunk: Buffer) => {
        stderr += chunk.toString();
      });

      process.on("error", (error) => {
        reject(
          new Error(
            `Failed to start Whisper: ${error.message}`,
          ),
        );
      });

      process.on("close", async (code) => {
        if (code !== 0) {
          reject(
            new Error(
              `Whisper exited with code ${code}: ${stderr.trim()}`,
            ),
          );

          return;
        }

        try {
          const outputFile = `${outputPrefix}.txt`;
          const text = await fs.readFile(
            outputFile,
            "utf8",
          );

          resolve(text);
        } catch (error) {
          reject(
            new Error(
              `Failed to read Whisper output: ${
                error instanceof Error
                  ? error.message
                  : String(error)
              }`,
            ),
          );
        }
      });
    });
  }
}