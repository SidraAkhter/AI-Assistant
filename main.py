from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
import shutil
import uuid
import os

from loader import load_pdf
from rag import ingest_document, ask_question

# ---------------------------------------------------
# FASTAPI APP
# ---------------------------------------------------
app = FastAPI()

# ---------------------------------------------------
# CREATE UPLOAD FOLDER
# ---------------------------------------------------
os.makedirs("uploads", exist_ok=True)

# ---------------------------------------------------
# FRONTEND UI
# ---------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def home():

    return """

    <html>

    <head>

        <title>AI RAG Chatbot</title>

        <style>

            body{
                font-family: Arial;
                background:#0f172a;
                color:white;
                display:flex;
                justify-content:center;
            }

            .container{
                width:700px;
                margin-top:40px;
                background:#111827;
                padding:20px;
                border-radius:12px;
            }

            h2{
                text-align:center;
            }

            input[type="text"]{
                width:75%;
                padding:10px;
                border:none;
                border-radius:8px;
            }

            button{
                padding:10px 15px;
                border:none;
                border-radius:8px;
                background:#2563eb;
                color:white;
                cursor:pointer;
            }

            button:hover{
                background:#1d4ed8;
            }

            .box{
                margin-top:20px;
                background:#1f2937;
                padding:15px;
                border-radius:10px;
                min-height:100px;
                white-space:pre-wrap;
            }

        </style>

    </head>

    <body>

        <div class="container">

            <h2>🤖 AI RAG Chatbot</h2>

            <!-- PDF Upload -->

            <form
                action="/upload-pdf"
                method="post"
                enctype="multipart/form-data"
            >

                <input type="file" name="file"/>

                <button type="submit">
                    Upload PDF
                </button>

            </form>

            <br>

            <!-- Question Input -->

            <input
                id="q"
                type="text"
                placeholder="Ask your question..."
            />

            <button onclick="askQuestion()">
                Ask
            </button>

            <!-- Answer Box -->

            <div class="box" id="answer">
                Waiting for question...
            </div>

        </div>

        <script>

            async function askQuestion(){

                let q = document.getElementById("q").value;

                let box = document.getElementById("answer");

                box.innerHTML = "Thinking...";

                try {

                    let response = await fetch(
                        "/ask?q=" + encodeURIComponent(q)
                    );

                    let data = await response.json();

                    console.log(data);

                    // HANDLE ERROR
                    if(data.error){

                        box.innerText =
                            "ERROR:\\n\\n" +
                            data.error;

                        return;
                    }

                    // SHOW ANSWER
                    box.innerText =
                        "ANSWER:\\n\\n" +
                        data.answer +
                        "\\n\\nSOURCES:\\n" +
                        data.sources.join(", ");

                }

                catch(err){

                    console.log(err);

                    box.innerText =
                        "Server Error Occurred";

                }

            }

        </script>

    </body>

    </html>

    """


# ---------------------------------------------------
# UPLOAD PDF
# ---------------------------------------------------
@app.post("/upload-pdf")
def upload_pdf(file: UploadFile = File(...)):

    try:

        # VALIDATE FILE
        if not file.filename.endswith(".pdf"):

            return JSONResponse(
                status_code=400,
                content={
                    "error": "Only PDF files allowed"
                }
            )

        # GENERATE UNIQUE FILE NAME
        filename = str(uuid.uuid4()) + ".pdf"

        path = f"uploads/{filename}"

        # SAVE FILE
        with open(path, "wb") as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )

        # LOAD PDF TEXT
        text = load_pdf(path)

        # INGEST INTO VECTOR DB
        total_chunks = ingest_document(
            text,
            filename
        )

        return JSONResponse(

            status_code=200,

            content={
                "message":
                "PDF uploaded successfully",

                "chunks":
                total_chunks
            }
        )

    except Exception as e:

        return JSONResponse(

            status_code=500,

            content={
                "error": str(e)
            }
        )


# ---------------------------------------------------
# ASK QUESTION
# ---------------------------------------------------
@app.get("/ask")
def ask(q: str):

    try:

        result = ask_question(q)

        return JSONResponse(

            status_code=200,

            content=result
        )

    except Exception as e:

        return JSONResponse(

            status_code=500,

            content={
                "error": str(e)
            }
        )