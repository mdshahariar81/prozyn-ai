export type VoiceInput = {
  stream: MediaStream;
  context: AudioContext;
  analyser: AnalyserNode;
  getLevel: () => number;
  stop: () => void;
};

export async function startVoiceInput(): Promise<VoiceInput> {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("Microphone access is not supported.");
  }

  const stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    },
  });

  const AudioContextClass =
    window.AudioContext ??
    (
      window as typeof window & {
        webkitAudioContext?: typeof AudioContext;
      }
    ).webkitAudioContext;

  if (!AudioContextClass) {
    stream.getTracks().forEach((track) => track.stop());

    throw new Error(
      "AudioContext is not supported.",
    );
  }

  const context = new AudioContextClass();

  if (context.state === "suspended") {
    await context.resume();
  }

  const source =
    context.createMediaStreamSource(stream);

  const analyser = context.createAnalyser();

  analyser.fftSize = 256;
  analyser.smoothingTimeConstant = 0.72;

  source.connect(analyser);

  const data = new Uint8Array(
    analyser.fftSize,
  );

  const getLevel = () => {
    analyser.getByteTimeDomainData(data);

    let sum = 0;

    for (const value of data) {
      const normalized =
        (value - 128) / 128;

      sum += normalized * normalized;
    }

    const rms = Math.sqrt(
      sum / data.length,
    );

    return Math.min(1, rms * 4);
  };

  const stop = () => {
    stream
      .getTracks()
      .forEach((track) => track.stop());

    void context.close();
  };

  return {
    stream,
    context,
    analyser,
    getLevel,
    stop,
  };
}

export type VoiceRecording = {
  blob: Blob;
  mimeType: string;
};

/**
 * Records until the user stops speaking.
 *
 * Flow:
 * TALK
 * → LISTENING
 * → speech detected
 * → continuous silence
 * → recording automatically stops
 */
export function recordVoiceUntilSilence(
  voiceInput: VoiceInput,
  options: {
    maxDurationMs?: number;
    silenceDurationMs?: number;
    speechThreshold?: number;
    minimumSpeechMs?: number;
  } = {},
): Promise<VoiceRecording> {
  const {
    maxDurationMs = 15_000,
    silenceDurationMs = 1_200,
    speechThreshold = 0.055,
    minimumSpeechMs = 500,
  } = options;

  if (!voiceInput.stream.active) {
    return Promise.reject(
      new Error(
        "Microphone stream is not active.",
      ),
    );
  }

  if (
    typeof MediaRecorder === "undefined"
  ) {
    return Promise.reject(
      new Error(
        "MediaRecorder is not supported.",
      ),
    );
  }

  const mimeTypes = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/mp4",
  ];

  const mimeType =
    mimeTypes.find((type) =>
      MediaRecorder.isTypeSupported(type),
    ) ?? "";

  const recorder = mimeType
    ? new MediaRecorder(
        voiceInput.stream,
        { mimeType },
      )
    : new MediaRecorder(
        voiceInput.stream,
      );

  const chunks: Blob[] = [];

  return new Promise(
    (resolve, reject) => {
      let timeoutId: number | null =
        null;

      let animationFrameId:
        | number
        | null = null;

      let speechDetected = false;

      let speechStartedAt: number | null =
        null;

      let lastSpeechAt = performance.now();

      const startedAt =
        performance.now();

      const cleanup = () => {
        if (timeoutId !== null) {
          window.clearTimeout(timeoutId);
        }

        if (
          animationFrameId !== null
        ) {
          window.cancelAnimationFrame(
            animationFrameId,
          );
        }
      };

      const finishRecording = () => {
        if (
          recorder.state === "recording"
        ) {
          recorder.stop();
        }
      };

      const monitorAudio = () => {
        if (
          recorder.state !== "recording"
        ) {
          return;
        }

        const now =
          performance.now();

        const elapsed =
          now - startedAt;

        const level =
          voiceInput.getLevel();

        if (
          level >= speechThreshold
        ) {
          if (!speechDetected) {
            if (
              speechStartedAt === null
            ) {
              speechStartedAt = now;
            }

            if (
              now -
                speechStartedAt >=
              minimumSpeechMs
            ) {
              speechDetected = true;
              lastSpeechAt = now;
            }
          } else {
            lastSpeechAt = now;
          }
        }

        if (
          speechDetected &&
          now - lastSpeechAt >=
            silenceDurationMs
        ) {
          finishRecording();
          return;
        }

        if (
          elapsed >= maxDurationMs
        ) {
          finishRecording();
          return;
        }

        animationFrameId =
          window.requestAnimationFrame(
            monitorAudio,
          );
      };

      recorder.ondataavailable = (
        event,
      ) => {
        if (event.data.size > 0) {
          chunks.push(event.data);
        }
      };

      recorder.onerror = () => {
        cleanup();

        reject(
          new Error(
            "Failed to record microphone audio.",
          ),
        );
      };

      recorder.onstop = () => {
        cleanup();

        const finalMimeType =
          recorder.mimeType ||
          mimeType ||
          "audio/webm";

        resolve({
          blob: new Blob(chunks, {
            type: finalMimeType,
          }),
          mimeType: finalMimeType,
        });
      };

      recorder.start();

      timeoutId =
        window.setTimeout(
          finishRecording,
          maxDurationMs,
        );

      animationFrameId =
        window.requestAnimationFrame(
          monitorAudio,
        );
    },
  );
}