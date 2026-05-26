type ModelMessage = { role: "system" | "user" | "assistant"; content: string };

export function getNemotronApiKey() {
  return process.env.NVIDIA_API_KEY || process.env.NEMOTRON_API_KEY || "";
}

export function getNemotronModel() {
  return (
    process.env.NVIDIA_MODEL ||
    process.env.COSTSAVVY_AI_MODEL ||
    "nvidia/nemotron-3-super-120b-a12b"
  );
}

export async function callNemotron(messages: ModelMessage[], temperature = 0.2) {
  const apiKey = getNemotronApiKey();
  if (!apiKey) return null;

  const response = await fetch(
    process.env.NVIDIA_BASE_URL || "https://integrate.api.nvidia.com/v1/chat/completions",
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model: getNemotronModel(),
        temperature,
        messages,
      }),
    }
  );

  if (!response.ok) {
    throw new Error(`Nemotron request failed with status ${response.status}`);
  }

  const data = (await response.json()) as {
    choices?: Array<{ message?: { content?: string | null } }>;
  };
  return data.choices?.[0]?.message?.content?.trim() || null;
}

