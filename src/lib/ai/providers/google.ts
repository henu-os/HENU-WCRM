import { AiError, type ChatMessage, type ProviderResult } from '../types'
import { MAX_OUTPUT_TOKENS } from '../defaults'
import {
  mergeConsecutive,
  normalizeUsage,
  toNetworkError,
  type ProviderArgs,
} from './shared'

const GOOGLE_API_BASE = 'https://generativelanguage.googleapis.com/v1beta/models'

interface GoogleCandidate {
  content?: {
    parts?: { text?: string }[]
    role?: string
  }
  finishReason?: string
}

interface GoogleResponse {
  candidates?: GoogleCandidate[]
  usageMetadata?: {
    promptTokenCount?: number
    candidatesTokenCount?: number
    totalTokenCount?: number
  }
  error?: {
    code?: number
    message?: string
    status?: string
  }
}

/**
 * Gemini API requires strictly alternating roles ('user' and 'model')
 * that begin with 'user'.
 */
function normalizeForGoogle(messages: ChatMessage[]) {
  const merged = mergeConsecutive(messages)
  while (merged.length > 0 && merged[0].role === 'assistant') {
    merged.shift()
  }
  if (merged.length === 0) {
    return [
      {
        role: 'user',
        parts: [{ text: '(The customer has not sent a message yet.)' }],
      },
    ]
  }
  return merged.map((m) => ({
    role: m.role === 'assistant' ? 'model' : 'user',
    parts: [{ text: m.content }],
  }))
}

/**
 * Call Google AI Studio (Gemini) generateContent endpoint with the caller's key.
 * Returns the raw assistant text + token usage (handoff parsing happens in `generateReply`).
 */
export async function generateGoogle(args: ProviderArgs): Promise<ProviderResult> {
  const { apiKey, model, systemPrompt, messages, timeoutMs } = args
  const endpoint = `${GOOGLE_API_BASE}/${encodeURIComponent(model)}:generateContent?key=${encodeURIComponent(apiKey)}`

  const payload: Record<string, unknown> = {
    contents: normalizeForGoogle(messages),
    generationConfig: {
      maxOutputTokens: MAX_OUTPUT_TOKENS,
    },
  }

  if (systemPrompt && systemPrompt.trim()) {
    payload.systemInstruction = {
      parts: [{ text: systemPrompt.trim() }],
    }
  }

  let res: Response
  try {
    res = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(timeoutMs),
    })
  } catch (err) {
    throw toNetworkError(err)
  }

  if (!res.ok) {
    let detail = ''
    try {
      const errJson = (await res.json()) as GoogleResponse
      detail = errJson?.error?.message ?? ''
    } catch {
      // Non-JSON response
    }

    const status = res.status
    const lower = detail.toLowerCase()
    const isApiKeyError =
      status === 400 &&
      (lower.includes('api key') ||
        lower.includes('api_key') ||
        lower.includes('apikey'))

    const code =
      status === 401 || status === 403 || isApiKeyError
        ? 'invalid_key'
        : status === 429
          ? 'rate_limited'
          : 'provider_error'

    const base =
      code === 'invalid_key'
        ? 'Google AI Studio rejected the API key'
        : code === 'rate_limited'
          ? 'Google AI Studio rate limit reached'
          : `Google AI Studio API error (${status})`

    throw new AiError(detail ? `${base}: ${detail}` : base, {
      code,
      status: code === 'invalid_key' ? 401 : 502,
    })
  }

  const data = (await res.json().catch(() => null)) as GoogleResponse | null
  const text = data?.candidates?.[0]?.content?.parts
    ?.map((p) => p.text ?? '')
    .join('')
    .trim()

  if (!text) {
    throw new AiError('Google AI Studio returned an empty response.', {
      code: 'empty_response',
    })
  }

  const usage = normalizeUsage({
    prompt: data?.usageMetadata?.promptTokenCount,
    completion: data?.usageMetadata?.candidatesTokenCount,
    total: data?.usageMetadata?.totalTokenCount,
  })

  return { text, usage }
}
