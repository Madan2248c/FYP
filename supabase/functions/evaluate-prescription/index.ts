/**
 * Supabase Edge Function for AMR Prescription Validation Evaluation
 *
 * This function serves as the LLM-as-a-Judge API endpoint for GRPO training.
 * It evaluates model-generated prescription validations using Groq Llama.
 */

// @ts-ignore: Deno types are available in Supabase Edge Functions runtime
import { serve } from "https://deno.land/std@0.168.0/http/server.ts"

// @ts-ignore: Deno global is available in Supabase Edge Functions runtime
declare const Deno: {
  env: {
    get(key: string): string | undefined;
  };
};

// Evaluation criteria configuration
const EVALUATION_CRITERIA = {
  clinical_accuracy: {
    description: "Correctness of clinical reasoning and medical facts",
    scale: "1-5 (1=Incorrect medical facts, 5=Perfectly accurate)",
    weight: 0.25
  },
  guideline_adherence: {
    description: "Alignment with ICMR 2025 antimicrobial treatment guidelines",
    scale: "1-5 (1=Contradicts guidelines, 5=Perfect adherence)",
    weight: 0.25
  },
  reasoning_completeness: {
    description: "Coverage of all required reasoning steps",
    scale: "1-5 (1=Missing critical steps, 5=All steps covered)",
    weight: 0.20
  },
  safety_awareness: {
    description: "Identification of allergies, contraindications, drug interactions",
    scale: "1-5 (1=Ignores safety, 5=Comprehensive safety analysis)",
    weight: 0.15
  },
  decision_appropriateness: {
    description: "Correctness of final decision (Approve/Modify/Reject)",
    scale: "1-5 (1=Wrong decision, 5=Perfect decision)",
    weight: 0.10
  },
  reference_accuracy: {
    description: "Accuracy of ICMR guideline page citations",
    scale: "1-5 (1=No/wrong citations, 5=Perfect citations)",
    weight: 0.05
  }
}

const PRIORITY_METRICS = [
  "clinical_accuracy",
  "guideline_adherence",
  "reasoning_completeness"
]

/**
 * Call Groq API to evaluate a single metric
 */
async function evaluateSingleMetric(
  metric: string,
  patientCase: any,
  modelOutput: string,
  groundTruth: string,
  groqApiKey: string
): Promise<any> {
  const criteria = EVALUATION_CRITERIA[metric as keyof typeof EVALUATION_CRITERIA]
  
  const prompt = `You are an expert medical evaluator specializing in antimicrobial stewardship and ICMR 2025 guidelines.

**EVALUATION TASK:**
Evaluate the quality of a model-generated prescription validation against a reference answer.

**METRIC TO EVALUATE:** ${metric}
**Description:** ${criteria.description}
**Scale:** ${criteria.scale}

**PATIENT CASE:**
${JSON.stringify(patientCase, null, 2)}

**MODEL OUTPUT (to evaluate):**
${modelOutput}

**REFERENCE ANSWER (ground truth):**
${groundTruth}

**INSTRUCTIONS:**
1. Carefully compare the MODEL OUTPUT against the REFERENCE ANSWER
2. Evaluate ONLY the "${metric}" aspect based on the criteria above
3. Assign a score from 1-5 (integer only)
4. Provide a brief justification (2-3 sentences)

**OUTPUT FORMAT (JSON only):**
{
  "metric": "${metric}",
  "score": <integer 1-5>,
  "justification": "<brief explanation>"
}

Respond with ONLY the JSON object, no additional text.`

  try {
    const response = await fetch("https://api.groq.com/openai/v1/chat/completions", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${groqApiKey}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        model: "llama-3.3-70b-versatile",
        messages: [
          {
            role: "system",
            content: "You are a medical evaluation expert. Respond only with valid JSON."
          },
          {
            role: "user",
            content: prompt
          }
        ],
        temperature: 0.1,
        max_tokens: 500,
        response_format: { type: "json_object" }
      })
    })

    if (!response.ok) {
      throw new Error(`Groq API error: ${response.statusText}`)
    }

    const data = await response.json()
    const result = JSON.parse(data.choices[0].message.content)
    
    // Validate score
    if (result.score < 1 || result.score > 5) {
      throw new Error(`Invalid score: ${result.score}`)
    }
    
    return result
    
  } catch (error) {
    console.error(`Error evaluating ${metric}:`, error)
    return {
      metric,
      score: 1,
      justification: `Evaluation failed: ${error.message}`
    }
  }
}

/**
 * Compute weighted reward from evaluation results
 */
function computeWeightedReward(evaluations: any): number {
  let totalReward = 0.0
  let totalWeight = 0.0
  
  for (const [metric, result] of Object.entries(evaluations)) {
    const criteria = EVALUATION_CRITERIA[metric as keyof typeof EVALUATION_CRITERIA]
    if (criteria) {
      // Normalize score from 1-5 to 0-1
      const normalizedScore = ((result as any).score - 1) / 4.0
      totalReward += normalizedScore * criteria.weight
      totalWeight += criteria.weight
    }
  }
  
  return totalWeight > 0 ? totalReward / totalWeight : 0.0
}

/**
 * Main handler function
 */
serve(async (req) => {
  // Handle CORS preflight
  if (req.method === "OPTIONS") {
    return new Response(null, {
      status: 204,
      headers: {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization"
      }
    })
  }

  try {
    // Get Groq API key from environment
    const groqApiKey = Deno.env.get("GROQ_API_KEY")
    if (!groqApiKey) {
      throw new Error("GROQ_API_KEY not configured")
    }

    // Parse request body
    const body = await req.json()
    const {
      patient_case,
      model_output,
      ground_truth,
      metrics = PRIORITY_METRICS
    } = body

    // Validate inputs
    if (!model_output || !ground_truth) {
      return new Response(
        JSON.stringify({
          success: false,
          error: "Missing required fields: model_output and ground_truth"
        }),
        {
          status: 400,
          headers: {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*"
          }
        }
      )
    }

    // Evaluate all requested metrics
    const evaluations: any = {}
    
    for (const metric of metrics) {
      const result = await evaluateSingleMetric(
        metric,
        patient_case,
        model_output,
        ground_truth,
        groqApiKey
      )
      evaluations[metric] = result
    }

    // Compute weighted reward
    const weightedReward = computeWeightedReward(evaluations)

    // Return results
    return new Response(
      JSON.stringify({
        success: true,
        evaluations,
        weighted_reward: weightedReward,
        metrics_evaluated: metrics.length,
        model: "llama-3.3-70b-versatile"
      }),
      {
        status: 200,
        headers: {
          "Content-Type": "application/json",
          "Access-Control-Allow-Origin": "*"
        }
      }
    )

  } catch (error) {
    console.error("Error:", error)
    return new Response(
      JSON.stringify({
        success: false,
        error: error.message
      }),
      {
        status: 500,
        headers: {
          "Content-Type": "application/json",
          "Access-Control-Allow-Origin": "*"
        }
      }
    )
  }
})

