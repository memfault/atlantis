package events

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/runatlantis/atlantis/server/logging"
)

const (
	openRouterURL             = "https://openrouter.ai/api/v1/chat/completions"
	openRouterAPIKeyEnv       = "OPENROUTER_API_KEY"
	openRouterSystemPromptEnv = "OPENROUTER_TERRAFORM_PLAN_SUMMARIZER_SYSTEM_PROMPT"
	openRouterTimeout         = 30 * time.Second
	defaultSystemPrompt       = "You are giving a summary of the changes in this terraform plan to a senior engineer. They are looking to know at a glance what is in this plan. Especially highlight any differences between environments; this is very important. For example, if a change is only being applied to one environment this MUST be called out. Your output should be a one-sentence summary followed by detailed bullet points of the changes to be made. Use as many bullet points as you need; the bullet points must cover every change. You may summarize a change, such as \"the AMI is being updated from X to Y in all environments\"; these would not need to be individual bullets. If a change is happening to every environment in the output, do not enumerate environments, just say \"all environments\" or \"all worker_generic\" environments."
)

// openRouterRequest represents the request payload for OpenRouter API
type openRouterRequest struct {
	Model    string              `json:"model"`
	Messages []openRouterMessage `json:"messages"`
}

// openRouterMessage represents a message in the chat completion request
type openRouterMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// openRouterResponse represents the response from OpenRouter API
type openRouterResponse struct {
	Choices []openRouterChoice `json:"choices"`
	Error   *openRouterError   `json:"error,omitempty"`
}

// openRouterChoice represents a choice in the response
type openRouterChoice struct {
	Message openRouterMessage `json:"message"`
}

// openRouterError represents an error from OpenRouter API
type openRouterError struct {
	Message string `json:"message"`
	Type    string `json:"type"`
}

// SummarizePlans sends Terraform plan outputs to OpenRouter for summarization.
// It combines all plan outputs into a single request and returns the summary.
// If the API key is not set or an error occurs, it returns an empty string
// and logs the error (fails gracefully).
func SummarizePlans(terraformOutputs []string, logger logging.SimpleLogging) string {
	if len(terraformOutputs) == 0 {
		logger.Debug("no terraform outputs to summarize")
		return ""
	}

	apiKey := os.Getenv(openRouterAPIKeyEnv)
	if apiKey == "" {
		logger.Debug("OPENROUTER_API_KEY not set, skipping plan summarization")
		return ""
	}

	// Combine all plan outputs with separators
	combinedOutput := strings.Join(terraformOutputs, "\n\n---\n\n")

	// Get system prompt from environment variable, with fallback to default
	systemPrompt := os.Getenv(openRouterSystemPromptEnv)
	if systemPrompt == "" {
		systemPrompt = defaultSystemPrompt
	}

	// Prepare the request
	reqBody := openRouterRequest{
		Model: "anthropic/claude-sonnet-4.5",
		Messages: []openRouterMessage{
			{
				Role:    "system",
				Content: systemPrompt,
			},
			{
				Role:    "user",
				Content: combinedOutput,
			},
		},
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		logger.Warn("failed to marshal OpenRouter request: %s", err)
		return ""
	}

	// Create HTTP request
	req, err := http.NewRequest("POST", openRouterURL, bytes.NewBuffer(jsonData))
	if err != nil {
		logger.Warn("failed to create OpenRouter request: %s", err)
		return ""
	}

	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", apiKey))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("HTTP-Referer", "https://github.com/memfault/atlantis-openrouter-summarizer")

	// Create HTTP client with timeout
	client := &http.Client{
		Timeout: openRouterTimeout,
	}

	// Send request
	logger.Debug("sending plan to OpenRouter for summarization")
	resp, err := client.Do(req)
	if err != nil {
		logger.Warn("failed to send request to OpenRouter: %s", err)
		return ""
	}
	defer resp.Body.Close()

	// Read response body
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		logger.Warn("failed to read OpenRouter response: %s", err)
		return ""
	}

	// Check HTTP status
	if resp.StatusCode != http.StatusOK {
		logger.Warn("OpenRouter API returned status %d: %s", resp.StatusCode, string(body))
		return ""
	}

	// Parse response
	var openRouterResp openRouterResponse
	if err := json.Unmarshal(body, &openRouterResp); err != nil {
		logger.Warn("failed to parse OpenRouter response: %s", err)
		return ""
	}

	// Check for API errors
	if openRouterResp.Error != nil {
		logger.Warn("OpenRouter API error: %s (type: %s)", openRouterResp.Error.Message, openRouterResp.Error.Type)
		return ""
	}

	// Extract summary from response
	if len(openRouterResp.Choices) == 0 {
		logger.Warn("OpenRouter response contained no choices")
		return ""
	}

	summary := strings.TrimSpace(openRouterResp.Choices[0].Message.Content)
	if summary == "" {
		logger.Warn("OpenRouter returned empty summary")
		return ""
	}

	logger.Debug("successfully received summary from OpenRouter")
	return summary
}
