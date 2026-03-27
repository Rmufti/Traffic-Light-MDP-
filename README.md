# 🚦 Traffic Light RL Optimization

This project uses Reinforcement Learning (Q-Learning) to optimize traffic light timings using the **SUMO** (Simulation of Urban MObility) engine and the **sumo-rl** Gymnasium wrapper.

---

## 🛠️ Prerequisites

Before running the project, ensure you have the following installed on your system:

1. **SUMO (Simulation of Urban MObility)**
   - **macOS:** `brew install sumo`
   - **Linux:** `sudo apt-get install sumo sumo-tools sumo-doc`
   - **Windows:** Download the installer from the [SUMO website](https://eclipse.dev/sumo/).

2. **Python 3.10+**
   - It is highly recommended to use a virtual environment.

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd Traffic_Light
```

## 2. Set up Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 3.Environment Variables
You must tell your system where SUMO is located. Add this to your .zshrc or .bashrc, or run it in your terminal before executing scripts:

```Bash
export SUMO_HOME=$(which sumo | sed 's|/bin/sumo||')
# On macOS Homebrew, it's usually: /opt/homebrew/opt/sumo/share/sumo
```

## 4. Project Structure
networks/: Contains .net.xml (road maps) and .rou.xml (traffic flows).
src/main.py: The entry point to run simulations.
sumo-rl/: The library used to wrap SUMO into an RL-compatible environment.
