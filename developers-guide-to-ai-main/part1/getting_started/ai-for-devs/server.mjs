import express from "express";
import { Ollama } from "ollama";
import cors from "cors";

const app = express();

app.use(cors());
app.use(express.json({ limit: "10mb" }));

const ollama = new Ollama();

// --------------------------------------------------
// Conversation history
// --------------------------------------------------

const conversation = [];

// --------------------------------------------------
// Ask AI
// --------------------------------------------------

app.post("/", async (request, response) => {
    response.type("text/plain");

    const body = request.body;

    if (!body.question) {
        response.status(400).send("No question was provided.");
        return;
    }

    conversation.push({
        role: "user",
        content: body.question
    });

    try {
        const streamIterator = await ollama.chat({
            model: "gemma4:e4b",

            messages: conversation,

            stream: true,

            options: {
                num_ctx: 32768
            }
        });

        let assistantResponse = "";

        for await (const chunk of streamIterator) {
            const text = chunk.message?.content || "";

            assistantResponse += text;
            response.write(text);
        }

        conversation.push({
            role: "assistant",
            content: assistantResponse
        });

        response.end();

    } catch (error) {
        console.error(error);

        // Remove the user message we just added so the conversation stays clean
        if (conversation.length && conversation[conversation.length - 1].role === "user") {
            conversation.pop();
        }

        response.status(500).send(
            "An error occurred while communicating with Ollama."
        );
    }
});

// --------------------------------------------------
// Generate downloadable file
// --------------------------------------------------

app.post("/generate-file", async (request, response) => {

    const { question, filename, fileType } = request.body;

    if (!question) {
        response.status(400).send("No file request was provided.");
        return;
    }

    if (!filename) {
        response.status(400).send("No filename was provided.");
        return;
    }

    if (!fileType) {
        response.status(400).send("No file type was provided.");
        return;
    }

    try {

        const filePrompt = `
You are a file-generation assistant.

Create the contents of the requested ${fileType} file.

User request:
${question}

IMPORTANT:
Return ONLY the contents of the file.

Do NOT include:
- Markdown code fences
- Explanations
- Comments about the file
- "Here is your file"
- Any text before or after the file contents

The response must be directly usable as a ${fileType} file.
`;

        const result = await ollama.chat({
            model: "gemma4:e4b",

            messages: [
                {
                    role: "user",
                    content: filePrompt
                }
            ],

            stream: false,

            options: {
                num_ctx: 32768
            }
        });

        let fileContent = result.message?.content || "";

        // Remove accidental Markdown code fences
        fileContent = fileContent
            .replace(/^```[a-zA-Z0-9_-]*\s*\n?/, "")
            .replace(/\n?```\s*$/, "");

        let mimeType = "text/plain";

        switch (fileType.toLowerCase()) {
            case "csv":
                mimeType = "text/csv";
                break;
            case "json":
                mimeType = "application/json";
                break;
            case "txt":
            case "text":
                mimeType = "text/plain";
                break;
            case "md":
            case "markdown":
                mimeType = "text/markdown";
                break;
            case "html":
                mimeType = "text/html";
                break;
            case "xml":
                mimeType = "application/xml";
                break;
            default:
                mimeType = "text/plain";
        }

        response.setHeader(
            "Content-Type",
            `${mimeType}; charset=utf-8`
        );

        response.setHeader(
            "Content-Disposition",
            `attachment; filename="${filename}"`
        );

        response.send(fileContent);

    } catch (error) {
        console.error(error);
        response.status(500).send(
            "An error occurred while generating the file."
        );
    }
});

// --------------------------------------------------
// Start a new conversation
// --------------------------------------------------

app.post("/reset", (request, response) => {
    conversation.length = 0;

    response.json({
        message: "Conversation history cleared."
    });
});

// --------------------------------------------------
// Start server
// --------------------------------------------------

app.listen(8000, () => {
    console.log("Server is running on port 8000");
});
