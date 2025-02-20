"use client"

import type React from "react"

import { useState } from "react"
import Image from "next/image"
import { useDropzone } from 'react-dropzone'
import { IoArrowBackOutline } from "react-icons/io5";
import { z } from "zod"
import { recipeSchema } from "@/schema/recipe"

export default function UploadForm() {
  const [image, setImage] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)

  const [result, setResult] = useState<{dishName: string, recipe: z.infer<typeof recipeSchema> } | null>(null)

  const handleFileChange = (files: File[]) => {
    const file = files[0]
    if (file) {
      setImage(file)
      setPreview(URL.createObjectURL(file))
    }
  }

  const {getRootProps, getInputProps} = useDropzone({onDrop: handleFileChange, maxFiles: 1, accept: {'image/*': []}})

  const [error, setError] = useState<Error | null>(null)

  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    if (!image) return

    const formData = new FormData()
    formData.append("image", image)

    setLoading(true)
    try {
      const response = await fetch("/api/generate-recipe", {
        method: "POST",
        body: formData,
      })

      if (response.ok) {
        const data = await response.json()
        setResult(data)
        setError(null)
      } else {
        console.error("Failed to generate recipe")
        setError(new Error('Failed to generate recipe'))
      }
    } catch (error) {
      console.error("Error:", error)
      setError(error as Error)
    } finally {
      setLoading(false)
    }
  }

  if (result) {
    return (
      <div className="mb-32 grid text-center lg:max-w-5xl lg:w-full lg:mb-0 lg:text-left">
        <div className="min-h-[600px]">
          <button className="flex items-center gap-2 mb-3 bg-amber-500 hover:bg-amber-600 text-white px-3 py-2 font-medium rounded-md" onClick={() => {
            setPreview(null)
            setResult(null)
            setError(null)
          }}>
            <IoArrowBackOutline className="mb-[1px]" />
            Upload a new image
          </button>
          <div className="mb-4">
            <Image src={preview || "/placeholder.svg"} alt="Preview" width={300} height={300} className="rounded-lg" />
          </div>
          <h2 className="text-3xl font-bold mb-1">{result.dishName}</h2>
          <div className="text-slate-500 mb-3">A recipe by Recipise.ai</div>
          <h3 className="font-bold mb-1 text-lg">Ingredients</h3>
          <ul className="mb-3">
            {result.recipe.ingredients.map(ingredient => (
              <li key={ingredient.name}>{ingredient.quantity} {ingredient.units} <strong>{ingredient.name}</strong></li>
            ))}
          </ul>
          <h3 className="font-bold mb-2 text-lg">Method</h3>
          <ul>
            {result.recipe.steps.map(step => (
              <li key={step.step} className="mb-1">- {step.step}</li>
            ))}
          </ul>
        </div>
      </div>
    )
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div className="flex items-center justify-center w-full">
        <div
          className="flex flex-col items-center justify-center w-full h-64 border-2 border-gray-300 border-dashed rounded-lg cursor-pointer bg-gray-50 dark:hover:bg-bray-800 dark:bg-gray-700 hover:bg-gray-100 dark:border-gray-600 dark:hover:border-gray-500 dark:hover:bg-gray-600"
          {...getRootProps()}
        >
          <div className="flex flex-col items-center justify-center pt-5 pb-6">
            <svg
              className="w-8 h-8 mb-4 text-gray-500 dark:text-gray-400"
              aria-hidden="true"
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 20 16"
            >
              <path
                stroke="currentColor"
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M13 13h3a3 3 0 0 0 0-6h-.025A5.56 5.56 0 0 0 16 6.5 5.5 5.5 0 0 0 5.207 5.021C5.137 5.017 5.071 5 5 5a4 4 0 0 0 0 8h2.167M10 15V6m0 0L8 8m2-2 2 2"
              />
            </svg>
            <p className="mb-2 text-sm text-gray-500 dark:text-gray-400">
              <span className="font-semibold">Click to upload</span> or drag and drop
            </p>
            <p className="text-xs text-gray-500 dark:text-gray-400">PNG, JPG or GIF (MAX. 800x400px)</p>
          </div>
          <input {...getInputProps()} type="file" className="hidden" accept="image/*" />
        </div>
      </div>
      {preview && (
        <div className="mt-4">
          <Image src={preview || "/placeholder.svg"} alt="Preview" width={300} height={300} className="rounded-lg" />
        </div>
      )}
      {error && (
        <div className="mt-4 bg-red-200 px-4 py-3 rounded-md">Something went wrong :(</div>
      )}
      <button
        type="submit"
        className="px-4 py-2 bg-amber-500 text-white rounded-lg hover:bg-amber-600 transition-colors flex items-center gap-2"
        disabled={!image}
      >
        {loading && <span className="loader" />}
        {loading ? 'Synthesising your recipe...' : 'Get Recipe'}
      </button>
    </form>
  )
}

