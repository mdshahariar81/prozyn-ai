import {
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import ProzynCore from "./components/ProzynCore";
import {
  recordVoiceUntilSilence,
  startVoiceInput,
  type VoiceInput,
} from "./services/voice";
import { sendMessage } from "./services/ai";
import "./App.css";

type Persona = "sara" | "alex";

type CoreState =
  | "idle"
  | "listening"
  | "thinking"
  | "speaking";

type ChatMessage = {
  id: number;
  role: "user" | "assistant";
  content: string;
};

const stateLabels: Record<CoreState, string> = {
  idle: "READY",
  listening: "LISTENING",
  thinking: "THINKING",
  speaking: "SPEAKING",
};

const personaInfo: Record<
  Persona,
  {
    name: string;
    role: string;
    description: string;
  }
> = {
  sara: {
    name: "SARA",
    role: "Companion",
    description:
      "Calm, caring, and naturally conversational.",
  },
  alex: {
    name: "ALEX",
    role: "Professional",
    description:
      "Focused, precise, and built for work.",
  },
};

const systemStatus = [
  ["AI CORE", "ONLINE"],
  ["MEMORY", "LOCAL"],
  ["VOICE", "READY"],
  ["NETWORK", "SECURE"],
];

const BACKEND_URL = "http://127.0.0.1:3001";

function App() {
  const [persona, setPersona] =
    useState<Persona>("sara");

  const [coreState, setCoreState] =
    useState<CoreState>("idle");

  const [chatOpen, setChatOpen] =
    useState(false);

  const [activityTick, setActivityTick] =
    useState(0);

  const [chatMessages, setChatMessages] =
    useState<ChatMessage[]>([]);

  const voiceInputRef =
    useRef<VoiceInput | null>(null);

  const recordingPromiseRef =
    useRef<
      ReturnType<
        typeof recordVoiceUntilSilence
      > | null
    >(null);

  const messageIdRef =
    useRef(0);

  useEffect(() => {
    const interval = window.setInterval(() => {
      setActivityTick(
        (current) => current + 1,
      );
    }, 1200);

    return () =>
      window.clearInterval(interval);
  }, []);

  useEffect(() => {
    return () => {
      voiceInputRef.current?.stop();
      voiceInputRef.current = null;
      recordingPromiseRef.current = null;
    };
  }, []);

  const activePersona =
    personaInfo[persona];

  const stopVoiceInput = useCallback(() => {
    voiceInputRef.current?.stop();
    voiceInputRef.current = null;
  }, []);

  const transcribeVoice = async (
    blob: Blob,
    mimeType: string,
  ): Promise<string> => {
    const formData = new FormData();

    const extension =
      mimeType.includes("mp4")
        ? "mp4"
        : "webm";

    const audioFile = new File(
      [blob],
      `prozyn-voice.${extension}`,
      {
        type: mimeType,
      },
    );

    formData.append(
      "file",
      audioFile,
    );

    const response = await fetch(
      `${BACKEND_URL}/voice/transcribe`,
      {
        method: "POST",
        body: formData,
      },
    );

    if (!response.ok) {
      const errorText =
        await response.text();

      throw new Error(
        `Transcription failed (${response.status}): ${errorText}`,
      );
    }

    const data =
      (await response.json()) as {
        text: string;
      };

    return data.text.trim();
  };

  const addChatMessage = useCallback(
    (
      role: ChatMessage["role"],
      content: string,
    ) => {
      setChatMessages((current) => [
        ...current,
        {
          id: messageIdRef.current++,
          role,
          content,
        },
      ]);
    },
    [],
  );

  const processAIMessage = useCallback(
    async (text: string) => {
      const message = text.trim();

      if (!message) {
        return;
      }

      addChatMessage("user", message);
      setCoreState("thinking");

      try {
        console.log(
          "PROZYN sending to AI:",
          message,
        );

        const result =
          await sendMessage(message);

        console.log(
          "PROZYN AI response:",
          result.response,
        );

        addChatMessage(
          "assistant",
          result.response,
        );

        /*
         * TTS is not connected yet.
         * Once TTS is added, this state will
         * become "speaking" while SARA/ALEX talks.
         */
        setCoreState("idle");
      } catch (error) {
        console.error(
          "Unable to send message to AI:",
          error,
        );

        addChatMessage(
          "assistant",
          "I couldn't connect to the local AI core. Please make sure the PROZYN backend and Ollama are running.",
        );

        setCoreState("idle");
      }
    },
    [addChatMessage],
  );

  const toggleListening =
    async () => {
      /*
       * If PROZYN is already listening,
       * do nothing here.
       *
       * The recorder automatically detects
       * when the user stops speaking.
       */
      if (
        coreState === "listening"
      ) {
        return;
      }

      /*
       * Don't start another recording while
       * PROZYN is processing or speaking.
       */
      if (
        coreState === "thinking" ||
        coreState === "speaking"
      ) {
        return;
      }

      try {
        const voiceInput =
          await startVoiceInput();

        voiceInputRef.current =
          voiceInput;

        setCoreState("listening");

        const recordingPromise =
          recordVoiceUntilSilence(
            voiceInput,
            {
              maxDurationMs: 15_000,
              silenceDurationMs: 1_200,
              speechThreshold: 0.055,
              minimumSpeechMs: 500,
            },
          );

        recordingPromiseRef.current =
          recordingPromise;

        const recording =
          await recordingPromise;

        recordingPromiseRef.current =
          null;

        /*
         * User stopped speaking.
         * Move to THINKING before Whisper.
         */
        setCoreState("thinking");

        stopVoiceInput();

        const text =
          await transcribeVoice(
            recording.blob,
            recording.mimeType,
          );

        console.log(
          "PROZYN voice transcription:",
          text,
        );

        if (!text) {
          console.warn(
            "Whisper returned an empty transcription.",
          );

          setCoreState("idle");
          return;
        }

        /*
         * Open the chat panel and show the
         * recognized message.
         */
        setChatOpen(true);

        /*
         * Send the Whisper transcription to
         * the local AI core / Ollama.
         */
        await processAIMessage(text);
      } catch (error) {
        console.error(
          "Unable to process voice input:",
          error,
        );

        recordingPromiseRef.current =
          null;

        stopVoiceInput();
        setCoreState("idle");
      }
    };

  const getAudioLevel =
    useCallback(() => {
      return (
        voiceInputRef.current?.getLevel() ??
        0
      );
    }, []);

  const cycleState = () => {
    setCoreState((current) => {
      if (current === "idle") {
        return "listening";
      }

      if (
        current === "listening"
      ) {
        return "thinking";
      }

      if (
        current === "thinking"
      ) {
        return "speaking";
      }

      return "idle";
    });
  };

  const changePersona = (
    nextPersona: Persona,
  ) => {
    if (
      nextPersona === persona
    ) {
      return;
    }

    stopVoiceInput();

    recordingPromiseRef.current =
      null;

    setPersona(nextPersona);
    setCoreState("idle");
  };

  return (
    <main
      className={`prozyn-app persona-${persona} state-${coreState}`}
      data-activity={activityTick}
    >
      <div className="ambient ambient-primary" />
      <div className="ambient ambient-secondary" />
      <div className="ambient ambient-tertiary" />

      <div
        className="star-field"
        aria-hidden="true"
      />

      <div
        className="soft-grid"
        aria-hidden="true"
      />

      <div
        className="vignette"
        aria-hidden="true"
      />

      <header className="topbar">
        <div className="brand">
          <div
            className="brand-mark"
            aria-hidden="true"
          >
            <span />
            <span />
            <span />
          </div>

          <div className="brand-copy">
            <strong>
              PROZYN
            </strong>

            <span>
              PERSONAL INTELLIGENCE
            </span>
          </div>
        </div>

        <div className="top-status">
          <span className="online-dot" />

          <span>
            LOCAL CORE
          </span>

          <i />

          <span>
            SECURE
          </span>
        </div>

        <button
          className="settings-button"
          type="button"
          aria-label="Settings"
        >
          <svg
            viewBox="0 0 24 24"
            aria-hidden="true"
          >
            <path d="M12 8.5a3.5 3.5 0 1 0 0 7 3.5 3.5 0 0 0 0-7Z" />
            <path d="M19.2 13.1a1.8 1.8 0 0 0 0-2.2l-.7-.8.8-1.7-1.8-1.8-1.7.8-.8-.7a1.8 1.8 0 0 0-2.2 0l-.9.6-.2-1.7H9.2L9 7.3l-.9-.6a1.8 1.8 0 0 0-2.2 0L4.1 8.5l.8 1.7-.7.8a1.8 1.8 0 0 0 0 2.2l.7.8-.8 1.7 1.8 1.8 1.7-.8.9.7.2 1.8h2.5l.2-1.8.9-.7 1.7.8 1.8-1.8-.8-1.7.7-.8Z" />
          </svg>
        </button>
      </header>

      <aside className="side-panel side-panel-left">
        <div className="panel-label">
          SYSTEM STATUS
        </div>

        <div className="status-list">
          {systemStatus.map(
            ([label, value]) => (
              <div
                className="status-row"
                key={label}
              >
                <span className="status-name">
                  <i />
                  {label}
                </span>

                <strong>
                  {value}
                </strong>
              </div>
            ),
          )}
        </div>

        <div className="activity-panel">
          <div className="activity-heading">
            <span>
              LIVE ACTIVITY
            </span>

            <i />
          </div>

          <div
            className="activity-wave"
            aria-hidden="true"
          >
            {Array.from({
              length: 18,
            }).map(
              (_, index) => (
                <span
                  key={index}
                  style={{
                    animationDelay: `${index * 70}ms`,
                  }}
                />
              ),
            )}
          </div>

          <div className="activity-state">
            <span className="mini-pulse" />

            {
              stateLabels[
                coreState
              ]
            }
          </div>
        </div>
      </aside>

      <aside className="side-panel side-panel-right">
        <div className="panel-label">
          CURRENT STATE
        </div>

        <div
          className="state-orb"
          aria-hidden="true"
        >
          <span />
          <span />
          <span />
        </div>

        <strong className="state-name">
          {
            stateLabels[
              coreState
            ]
          }
        </strong>

        <span className="state-description">
          {coreState === "idle"
            ? "Ready for your next instruction"
            : coreState === "listening"
              ? "Listening for your voice"
              : coreState ===
                  "thinking"
                ? "Processing your request"
                : "Preparing a response"}
        </span>

        <div className="state-line" />

        <div className="entity-readout">
          <span>
            ACTIVE ENTITY
          </span>

          <strong>
            {activePersona.name}
          </strong>

          <small>
            {activePersona.role}
          </small>
        </div>
      </aside>

      <section className="entity-stage">
        <div
          className="entity-aura aura-one"
          aria-hidden="true"
        />

        <div
          className="entity-aura aura-two"
          aria-hidden="true"
        />

        <div
          className="entity-ring ring-one"
          aria-hidden="true"
        />

        <div
          className="entity-ring ring-two"
          aria-hidden="true"
        />

        <div
          className="entity-ring ring-three"
          aria-hidden="true"
        />

        <div className="entity-state">
          <span />

          <strong>
            {
              stateLabels[
                coreState
              ]
            }
          </strong>
        </div>

        <button
          className="entity-button"
          type="button"
          aria-label="Interact with PROZYN"
          onClick={cycleState}
        >
          <ProzynCore
            persona={persona}
            state={coreState}
            getLevel={getAudioLevel}
          />
        </button>

        <div className="entity-identity">
          <span>
            PROZYN ENTITY
          </span>

          <strong>
            {activePersona.name}
          </strong>

          <small>
            {
              activePersona.description
            }
          </small>
        </div>

        <div
          className="entity-particles particle-set-one"
          aria-hidden="true"
        />

        <div
          className="entity-particles particle-set-two"
          aria-hidden="true"
        />
      </section>

      <section className="persona-dock">
        <div className="dock-heading">
          <span>
            AI COMPANION
          </span>

          <i />
        </div>

        <div className="persona-switch">
          <button
            type="button"
            className={
              persona === "sara"
                ? "active sara"
                : "sara"
            }
            onClick={() =>
              changePersona("sara")
            }
          >
            <span className="persona-dot" />

            <span className="persona-text">
              <strong>
                SARA
              </strong>

              <small>
                Companion
              </small>
            </span>
          </button>

          <button
            type="button"
            className={
              persona === "alex"
                ? "active alex"
                : "alex"
            }
            onClick={() =>
              changePersona("alex")
            }
          >
            <span className="persona-dot" />

            <span className="persona-text">
              <strong>
                ALEX
              </strong>

              <small>
                Professional
              </small>
            </span>
          </button>

          <div className="switch-indicator" />
        </div>
      </section>

      <div className="interaction-dock">
        <div className="interaction-status">
          <span className="interaction-dot" />

          <span>
            {
              stateLabels[
                coreState
              ]
            }
          </span>
        </div>

        <button
          className={`mic-button ${
            coreState === "listening"
              ? "active"
              : ""
          }`}
          type="button"
          aria-label={
            coreState === "listening"
              ? "Listening"
              : "Start listening"
          }
          aria-pressed={
            coreState === "listening"
          }
          onClick={() => {
            void toggleListening();
          }}
        >
          <svg
            viewBox="0 0 24 24"
            aria-hidden="true"
          >
            <rect
              x="8"
              y="3.5"
              width="8"
              height="12"
              rx="4"
            />

            <path d="M5 11.5a7 7 0 0 0 14 0M12 18.5V21M9 21h6" />
          </svg>
        </button>

        <button
          className={`chat-button ${
            chatOpen
              ? "active"
              : ""
          }`}
          type="button"
          aria-label="Open chat"
          onClick={() =>
            setChatOpen(
              (current) =>
                !current,
            )
          }
        >
          <svg
            viewBox="0 0 24 24"
            aria-hidden="true"
          >
            <path d="M5 6.5h14v10H9l-4 3v-13Z" />
            <path d="M8 10.5h8M8 13.5h5" />
          </svg>
        </button>
      </div>

      <div className="bottom-meta">
        <span>
          PROZYN / LOCAL INTELLIGENCE
        </span>

        <span>
          V0.1.0
        </span>
      </div>

      {chatOpen && (
        <aside className="chat-panel">
          <div className="chat-panel-header">
            <div>
              <span>
                PROZYN CHANNEL
              </span>

              <strong>
                {activePersona.name}
              </strong>
            </div>

            <button
              type="button"
              aria-label="Close chat"
              onClick={() =>
                setChatOpen(false)
              }
            >
              ×
            </button>
          </div>

          <div className="chat-panel-body">
            <div className="chat-avatar">
              <span />
            </div>

            <div>
              <span className="chat-kicker">
                LOCAL INTELLIGENCE
              </span>

              <p>
                {
                  activePersona.description
                }
              </p>
            </div>
          </div>

          {chatMessages.length > 0 && (
            <div className="chat-messages">
              {chatMessages.map(
                (message) => (
                  <div
                    key={message.id}
                    className={`chat-message ${message.role}`}
                  >
                    <span>
                      {message.role ===
                      "user"
                        ? "YOU"
                        : activePersona.name}
                    </span>

                    <p>
                      {message.content}
                    </p>
                  </div>
                ),
              )}
            </div>
          )}

          <div className="chat-input">
            <span>
              Ask{" "}
              {activePersona.name}{" "}
              anything...
            </span>

            <i />
          </div>
        </aside>
      )}
    </main>
  );
}

export default App;
