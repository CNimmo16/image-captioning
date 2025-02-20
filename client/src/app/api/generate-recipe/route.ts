import { recipeSchema } from "@/schema/recipe";
import { type NextRequest, NextResponse } from "next/server"
import OpenAI from "openai";
import { zodResponseFormat } from "openai/helpers/zod";

const openai = new OpenAI();

export async function POST(req: NextRequest) {
  try {
    const formData = await req.formData()
    const image = formData.get("image") as File

    if (!image) {
      return NextResponse.json({ error: "No image provided" }, { status: 400 })
    }

    const imageBuffer = await image.arrayBuffer()
    const base64Image = Buffer.from(imageBuffer).toString("base64")

    const { dish_name: dishName } = await fetch(`${process.env.SERVER_URL}/predict`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ image: base64Image }),
    }).then(res => res.json())

    const completion = await openai.beta.chat.completions.parse({
      model: "gpt-4o-2024-08-06",
      messages: [
        { role: "system", content: "You are a helpful assistant that generates a recipe for the dish name the user inputs" },
        { role: "user", content: dishName },
      ],
      response_format: zodResponseFormat(recipeSchema, "recipe"),
    });
  
    const recipe = completion.choices[0].message.parsed;

    return NextResponse.json({ dishName, recipe })
  } catch (error) {
    console.error("Error:", error)
    return NextResponse.json({ error: "Failed to generate recipe" }, { status: 500 })
  }
}
