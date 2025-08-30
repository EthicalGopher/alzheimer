package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/gofiber/fiber/v2"
	"github.com/gofiber/fiber/v2/middleware/logger"
	"github.com/google/uuid"
)

const (
	uploadDir = "./uploads"
	modelPath = "alzheimers_model.onnx"
)

// displayOrder defines the consistent order for presenting classification results in the UI.
var displayOrder = []string{
	"No Impairment",
	"Very Mild Impairment",
	"Mild Impairment",
	"Moderate Impairment",
}

// ClassificationResult holds the structured data from the Python script's JSON output.
type ClassificationResult struct {
	PredictedClass   string            `json:"predicted_class"`
	Confidence       string            `json:"confidence"`
	AllProbabilities map[string]string `json:"all_probabilities"`
}

func main() {
	if err := os.MkdirAll(uploadDir, os.ModePerm); err != nil {
		log.Fatalf("Could not create upload directory: %v", err)
	}

	app := fiber.New()
	app.Use(logger.New())

	// Serve static files from the "public" directory.
	app.Static("/", "./public")

	// The classification endpoint now calls the script with proper flags.
	app.Post("/classify", handleImageClassification)

	log.Println("Server is starting on http://localhost:3000")
	log.Fatal(app.Listen(":3000"))
}

func handleImageClassification(c *fiber.Ctx) error {
	file, err := c.FormFile("image")
	if err != nil {
		return c.Status(fiber.StatusBadRequest).SendString(
			`<p style="color: orange;">Please select an image file to upload.</p>`,
		)
	}

	ext := filepath.Ext(file.Filename)
	uniqueFilename := uuid.New().String() + ext
	savePath := filepath.Join(uploadDir, uniqueFilename)

	if err := c.SaveFile(file, savePath); err != nil {
		return c.Status(fiber.StatusInternalServerError).SendString(
			`<p style="color: red;">Error: Could not save file on server.</p>`,
		)
	}

	defer func() {
		if err := os.Remove(savePath); err != nil {
			log.Printf("Warning: failed to remove temp file %s: %v", savePath, err)
		}
	}()

	log.Printf("Executing classifier for image: %s", savePath)
	cmd := exec.Command("python", "python/classifier.py",
		"--model", modelPath,
		"--image", savePath,
		"--json", // Request clean JSON output
	)

	output, err := cmd.CombinedOutput()
	if err != nil {
		log.Printf("Python script execution failed: %v\nOutput from script: %s", err, string(output))
		return c.Status(fiber.StatusInternalServerError).SendString(
			`<p style="color: red;">Error during classification. Check server logs for details.</p>`,
		)
	}

	var result ClassificationResult
	if err := json.Unmarshal(output, &result); err != nil {
		log.Printf("Failed to parse JSON from python script: %v\nRaw output: %s", err, string(output))
		return c.Status(fiber.StatusInternalServerError).SendString(
			`<p style="color: orange;">Error: Could not understand the result from the classifier.</p>`,
		)
	}

	// Build the list of all probabilities for display
	var probsListBuilder strings.Builder
	// Use the package-level displayOrder slice for consistent presentation
	for _, key := range displayOrder {
		if val, ok := result.AllProbabilities[key]; ok {
			// Highlight the predicted class in the list
			if key == result.PredictedClass {
				probsListBuilder.WriteString(fmt.Sprintf(`<li><strong>%s: %s</strong></li>`, key, val))
			} else {
				probsListBuilder.WriteString(fmt.Sprintf(`<li>%s: %s</li>`, key, val))
			}
		}
	}

	// Create the final, decorated HTML snippet
	htmlResponse := fmt.Sprintf(`
        <h4>Classification Result</h4>
        <p><strong>Prediction:</strong> <mark>%s</mark></p>
        <p><strong>Confidence:</strong> %s</p>
        <details open>
            <summary>View All Probabilities</summary>
            <ul>
                %s
            </ul>
        </details>
    `, result.PredictedClass, result.Confidence, probsListBuilder.String())

	return c.SendString(htmlResponse)
}
