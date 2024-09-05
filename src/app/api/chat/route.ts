import { OpenAI } from 'openai'
import { Message, OpenAIStream, StreamingTextResponse } from 'ai'
import { getContext } from '@/utils/context'

// Create an OpenAI API client (edge-friendly!)
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
})

// IMPORTANT! Set the runtime to edge
export const runtime = 'edge'

export async function POST(req: Request) {
  try {
    const { messages } = await req.json()

    // Get the last message
    const lastMessage = messages[messages.length - 1]

    // Fetch context based on the content of the last message
    const context = await getContext(lastMessage.content, '')

    // System prompt defining the model's behavior and incorporating context for Texas ISD financial data
    const systemPrompt = `
      You are a financial assistant trained on Texas ISD financial history and policies, capable of understanding complex financial reports, trends, and budgetary data. You provide accurate, well-organized, and insightful answers to questions related to the funding, expenditure, and budgetary performance of school districts across Texas. You are also knowledgeable about historical budget allocations, revenue sources, tax implications, and how various factors such as enrollment and property values impact school district finances.

      Your goal is to help users understand the financial status and history of specific ISDs, including comparisons across different time periods and districts.

      START CONTEXT BLOCK
      ${context}
      END OF CONTEXT BLOCK

      Use the provided context to answer questions. If the context doesn't contain the answer, say "I don't have enough information to answer that question accurately." 
      Do not invent or assume information not present in the context. If you learn new information, incorporate it into your knowledge base for future responses.
    `

    // Key training questions for Pinecone, targeting financial history of Texas ISDs
    const trainingQuestions = [
      { role: 'system', content: systemPrompt },
      ...messages.filter((message: Message) => message.role === 'user')
    ]

    // Ask OpenAI for a streaming chat completion given the prompt
    const response = await openai.chat.completions.create({
      model: 'gpt-4',
      stream: true,
      messages: trainingQuestions
    })

    // Convert the OpenAI response into a stream for real-time updates
    const stream = OpenAIStream(response)
    
    // Send the stream response
    return new StreamingTextResponse(stream)
    
  } catch (error) {
    console.error('Error in chat completion:', error)
    return new Response('An error occurred during your request.', { status: 500 })
  }
}
