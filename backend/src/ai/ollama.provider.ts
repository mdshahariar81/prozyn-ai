import type {
  AIProvider,
  AIRequest,
  AIResponse,
} from "./ai.service.js";

type OllamaChatResponse = {
  model: string;
  message?: {
    role: string;
    content: string;
  };
};

export class OllamaProvider implements AIProvider {
  private readonly baseUrl = "http://127.0.0.1:11434";
  private readonly model = "qwen3:4b";

  async generate(request: AIRequest): Promise<AIResponse> {
    const response = await fetch(`${this.baseUrl}/api/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: this.model,
        messages: request.messages,
        stream: false,
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();

      throw new Error(
        `Ollama request failed (${response.status}): ${errorText}`,
      );
    }

    const data = (await response.json()) as OllamaChatResponse;

    return {
      content: data.message?.content?.trim() ?? "",
      model: data.model,
    };
  }
}
