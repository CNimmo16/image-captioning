import { z } from 'zod'

export const recipeSchema = z.object({
  ingredients: z.array(z.object({
    name: z.string(),
    units: z.string(),
    quantity: z.string(),
  })),
  steps: z.array(z.object({
    step: z.string(),
  }))
})
