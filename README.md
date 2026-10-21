# Kelvin6K — Autonomous AI Voice Assistant for 3D Construction Printing

An intelligent, low-latency (<3s) voice-to-voice factory floor assistant built for the **Kelvin6K 3D Concrete Printing Robot**. 

Powered by **Google Gemini (`gemini-flash-lite-latest`)**, a **SQLite Operational Database**, and a **Hybrid RAG Knowledge Base (BM25 + Semantic Vectors)**, the assistant acts like a real factory colleague—answering technical questions, managing operator shifts, tracking printing jobs & 9-hour overtime rules, logging floor issues, and learning new operational facts in real time.

---

## Key Features

- **100% Autonomous Gemini Brain**: No rigid keyword scripts. Understands natural speech, homophones, and open-ended questions.
- **Universal Database Read/Write (`kelvin6k.db`)**: Tracks operators, active shifts, attendance, printing jobs, wall production, machine state, and shift issues.
- **Automatic 9-Hour Overtime Engine**: Automatically calculates regular hours (`<= 9.0 hrs`) vs. overtime hours (`> 9.0 hrs`) for printing jobs and shifts.
- **Live RAG Knowledge Base & Self-Learning**: Searches Kelvin6K manuals, SOPs, and troubleshooting guides. When you tell the agent to remember a new fix or production note, it saves it to `docs/knowledge/learned_knowledge.md` and indexes it in the background (<10ms).
- **Zero-Latency Voice Pipeline**: Uses browser-native real-time Speech-to-Text (`Web Speech API`) + instant spoken synthesis (`speechSynthesis`), with automatic server-side `faster-whisper` fallback.

---

## Quick Start — Windows Setup (cmd or PowerShell)

**1. Clone the repository**
```cmd
git clone [https://github.com/mohammedhamdaan/kelvin-agent.git](https://github.com/mohammedhamdaan/kelvin-agent.git)
cd kelvin-agent
2. Create virtual environment & install Python libraries

DOS
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
3. Configure your .env file with your Gemini API Key

DOS
copy .env.example .env
echo.>> .env
echo GEMINI_API_KEY=YOUR_REAL_GEMINI_API_KEY>> .env
4. Install Frontend UI packages

DOS
cd ui
npm install
cd ..
5. Run the App (Open 2 Command Prompt Windows)

CMD Window 1 (Backend Server — Port 8000):

DOS
.venv\Scripts\activate.bat
python run_api.py --port 8000
CMD Window 2 (Frontend UI — Port 5173):

DOS
cd ui
npm run dev
Open http://localhost:5173 in Google Chrome or Microsoft Edge.

Quick Start — macOS / Linux Setup
1. Clone & install backend dependencies

Bash
git clone [https://github.com/mohammedhamdaan/kelvin-agent.git](https://github.com/mohammedhamdaan/kelvin-agent.git)
cd kelvin-agent
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
2. Configure your .env file

Bash
cp .env.example .env
(Open .env and set GEMINI_API_KEY=AIzaSy...)

3. Install Frontend dependencies

Bash
cd ui
npm install
cd ..
4. Run the App (Open 2 Terminal Tabs)

Terminal 1 (Backend — Port 8000): source .venv/bin/activate && python3 run_api.py --port 8000

Terminal 2 (Frontend — Port 5173): cd ui && npm run dev

Open http://localhost:5173 in Google Chrome.

Example Voice Commands
"Start the morning shift for Hamdan."

"Hamdan completed printing 2 walls today, save that."

"While printing the wall, the nozzle stopped. What could be the reason?"

"Save that this nozzle stop issue was faced by Hamdan today while printing Wall 2."

**3. Configure your `.env` file**
- Copy the template:
  ```cmd
  copy .env.example .env