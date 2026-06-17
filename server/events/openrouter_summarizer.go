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
	openRouterModelEnv        = "OPENROUTER_TERRAFORM_PLAN_SUMMARIZER_MODEL"
	openRouterTimeout         = 30 * time.Second
	defaultModel              = "anthropic/claude-opus-4.8"
	defaultSystemPrompt       = `You summarize Terraform plans for a senior engineer scanning a PR. Give the gist at a glance: a few thematic bullets, never a flat list of every resource.

If no project has resource changes, reply with exactly one line and stop:
"**No changes.** All {N} projects match current state."

Otherwise:
1. First line, bold: the combined totals - "**{X} to add, {Y} to change, {Z} to destroy across {M} of {N} projects.**"
2. Then a few bullets, each describing one logical change, NOT one resource. Collapse aggressively:
   - Many near-identical resources changing the same way become ONE bullet with a count, not individual lines: "7 listener rules created, one per project key"; "all 4 ASG launch templates rolled to new versions". Never enumerate the instances or print their IDs, keys, or priorities.
   - A coordinated change spanning several resource types but serving one purpose is ONE bullet: "reworked org2348 ALB routing: replaced 3 multi-key rules with 7 single-key rules".
   - The same change across environments is ONE bullet ending in its scope: "(all environments)", "(staging + production)".
   - Keep genuinely unrelated changes as separate bullets.
3. Say WHAT changed in plain terms; leave out the specific values. Drop opaque identifiers - AMI IDs, image hashes, ARNs, resource IDs, long version strings: "bumped the AMI" or "rolled the launch templates to a new AMI" is enough, no hash. Keep a value inline only when it is short and meaningful at a glance, like a capacity (10 -> 12), a port, or a count.
4. Stay complete: the bullets together must account for all X+Y+Z resources, but "account for" means summarize-with-a-count, never omit. Prefix the bullet with a warning sign emoji for any destroy or replace, and for anything that hits only some environments.
5. More than ~6 bullets means you are enumerating instead of summarizing - group harder.

Never report terraform init output, provider/module versions, or backend config - those are not resource changes.`
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

	// Get model from environment variable, with fallback to default
	model := os.Getenv(openRouterModelEnv)
	if model == "" {
		model = defaultModel
	}

	// Prepare the request
	reqBody := openRouterRequest{
		Model: model,
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
