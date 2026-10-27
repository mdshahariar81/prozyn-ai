import Fastify from "fastify";
import cors from "@fastify/cors";
import multipart from "@fastify/multipart";
import { spawn } from "node:child_process";
import { promises as fs } from "node:fs";
import os from "node:os";
import path from "node:path";

import { AIService } from "./ai/ai.service.js";
import { OllamaProvider } from "./ai/ollama.provider.js";
import { WhisperService } from "./voice/whisper.service.js";

const app = Fastify({
  logger: true,
});

await app.register(cors, {
  origin: [
    "http://localhost:1420",
    "http://127.0.0.1:1420",
  ],
});

await app.register(multipart, {
  limits: {
    fileSize: 10 * 1024 * 1024,
  },
});

const aiService = new AIService(new OllamaProvider());
const whisperService = new WhisperService();

const FFMPEG_PATH =
  process.env.FFMPEG_PATH ?? "/opt/homebrew/bin/ffmpeg";

app.get("/health", async () => {
  return {
    status: "ok",
    service: "PROZYN backend",
  };
});

app.post<{
  Body: {
    message: string;
  };
}>("/chat", async (request, reply) => {
  const message = request.body?.message?.trim();

  if (!message) {
    return reply.status(400).send({
      error: "Message is required.",
    });
  }

  try {
    const response = await aiService.generate({
      messages: [
        {
          role: "system",
          content:
            "You are PROZYN, a concise personal AI assistant. Answer clearly and helpfully.",
        },
        {
          role: "user",
          content: message,
        },
      ],
    });

    return {
      response: response.content,
      model: response.model,
    };
  } catch (error) {
    request.log.error(error);

    return reply.status(500).send({
      error: "Failed to generate AI response.",
    });
  }
});

app.post("/voice/transcribe", async (request, reply) => {
  let temporaryDirectory: string | null = null;

  try {
    const file = await request.file();

    if (!file) {
      return reply.status(400).send({
        error: "Audio file is required.",
      });
    }

    temporaryDirectory = await fs.mkdtemp(
      path.join(os.tmpdir(), "prozyn-audio-"),
    );

    const inputFilePath = path.join(
      temporaryDirectory,
      "input-audio",
    );

    const convertedWavPath = path.join(
      temporaryDirectory,
      "converted.wav",
    );

    const audioBuffer = await file.toBuffer();

    await fs.writeFile(
      inputFilePath,
      audioBuffer,
    );

    await convertAudioToWhisperWav(
      inputFilePath,
      convertedWavPath,
    );

    const result = await whisperService.transcribe(
      convertedWavPath,
    );

    return {
      text: result.text,
    };
  } catch (error) {
    request.log.error(error);

    return reply.status(500).send({
      error:
        error instanceof Error
          ? error.message
          : "Failed to transcribe audio.",
    });
  } finally {
    if (temporaryDirectory) {
      try {
        await fs.rm(temporaryDirectory, {
          recursive: true,
          force: true,
        });
      } catch (cleanupError) {
        request.log.error(
          cleanupError,
          "Failed to clean up temporary audio files.",
        );
      }
    }
  }
});

function convertAudioToWhisperWav(
  inputPath: string,
  outputPath: string,
): Promise<void> {
  return new Promise((resolve, reject) => {
    const ffmpeg = spawn(FFMPEG_PATH, [
      "-y",
      "-i",
      inputPath,
      "-ar",
      "16000",
      "-ac",
      "1",
      "-c:a",
      "pcm_s16le",
      outputPath,
    ]);

    let stderr = "";

    ffmpeg.stderr.on(
      "data",
      (chunk: Buffer) => {
        stderr += chunk.toString();
      },
    );

    ffmpeg.on("error", (error) => {
      reject(
        new Error(
          `Failed to start FFmpeg: ${error.message}`,
        ),
      );
    });

    ffmpeg.on("close", (code) => {
      if (code !== 0) {
        reject(
          new Error(
            `FFmpeg exited with code ${code}: ${stderr.trim()}`,
          ),
        );

        return;
      }

      resolve();
    });
  });
}

const start = async () => {
  try {
    await app.listen({
      port: 3001,
      host: "127.0.0.1",
    });
  } catch (error) {
    app.log.error(error);
    process.exit(1);
  }
};

void start();