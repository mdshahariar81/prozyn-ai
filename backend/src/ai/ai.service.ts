export type AIMessageRole = "system" | "user" | "assistant";

export type AIMessage = {
  role: AIMessageRole;
  content: string;
};

export type AIRequest = {
  messages: AIMessage[];
};

export type AIResponse = {
  content: string;
  model: string;
};

export interface AIProvider {
  generate(request: AIRequest): Promise<AIResponse>;
}

export class AIService {
  constructor(private readonly provider: AIProvider) {}

  async generate(request: AIRequest): Promise<AIResponse> {
    return this.provider.generate(request);
  }
}
