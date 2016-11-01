# gym-mountaincar-sample

### INSTALLATION
1. Review [requirements](http://docs.bons.ai/getting-started/lets-get-started) for installing the Bonsai CLI.
1. Install the Bonsai Command Line Interface (CLI).
       pip install bonsai-cli
       bonsai configure
1. Install the simulator's requirements:
       pip install -r requirements.txt

### HOW TO TRAIN YOUR BRAIN
1. If you haven't already created a BRAIN at the website, create one now:
       bonsai brain create <your_brain>
1. Load your Inkling file into your brain. Review our [Inkling Guide](http://docs.bons.ai/inkling-guide-pages/introduction) for help with Inkling.
       bonsai brain load <your_brain> mountaincar.ink
1. Enable training mode for your brain. Please note that training may take many hours.
       bonsai brain train start <your_brain>
1. Connect a simulator for training. For inspiration, check out our [Mountain Car demo](https://github.com/BonsaiAI/gym-mountaincar-sample).
       python mountaincar_simulator.py --train-brain=<your_brain> --headless
1. When training has hit a sufficient accuracy, disable training mode.
       bonsai brain train stop <your_brain>

### USE YOUR BRAIN

1. Run the simulator using predictions from your brain.
       python mountaincar_simulator.py --predict-brain=<your_brain> --predict-version=latest
