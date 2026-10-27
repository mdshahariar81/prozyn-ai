export type ChatResponse = {
  response: string;
  model: string;
};

const BACKEND_URL = "http://127.0.0.1:3001";

export async function sendMessage(
  message: string,
): Promise<ChatResponse> {
  const response = await fetch(`${BACKEND_URL}/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      message,
    }),
  });

  if (!response.ok) {
    const errorText = await response.text();

    throw new Error(
      `AI request failed (${response.status}): ${errorText}`,
    );
  }

  return (await response.json()) as ChatResponse;
}
