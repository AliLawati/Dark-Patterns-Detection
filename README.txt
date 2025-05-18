----------------------------------------------------
Welcome to the Dark Patterns Classification Project!
----------------------------------------------------
NOTE: All of the code in this project was written and run using PyCharm IDE.

Please follow the instructions written below before running any file in the project to make sure that everything works well.

1. Python and PyCharm Installation:
-----------------------------------
a. Install Python version 3.10.0 if you don't have it (https://www.python.org/downloads/release/python-3100/)
b. Run the installer and follow the instructions
c. Install PyCharm Community Edition if you don't have it (https://www.jetbrains.com/pycharm/download)
d. Run the installer and follow the instructions
e. Run PyCharm, open the project, and wait for it to load.
f. After the project is ready, click in the bottom right corner (where written <No Interpreter>)
     i. Select "Add New Interpreter"
    ii. Select "Add Local Interpreter"
   iii. Add your Python 3.10 interpreter as "Virtualenv" or "System Interpreter"
-----------------------------------

2. Required Dependencies Installation:
--------------------------------------
a. After the interpreter is ready, install the requirements using the cli in PyCharm IDE bottom left icon (terminal)
b. Type the command: pip install -r requirements.txt
--------------------------------------

3. Model Installation:
----------------------
a. Install ollama for the appropriate operating system, if you don't have it (https://ollama.com/)
b. Run the installer and follow the instructions
c. Run this command in cli: ollama run mistral
d. After the model is installed in the system, you can exit the cli by typing: /bye
----------------------

4. Running and Testing the API:
-------------------------------
a. Go to llm_model_api.py file
b. Run it by using one of the following 2 ways:
      i. Just run the Python file normally in the IDE or by using "python llm_model_api.py" command in the terminal
        (preferred)
     ii. Run the following command in your terminal: "uvicorn llm_model_api:app --reload" (this method might use a
        different path to access the endpoints, so you'll need to update the path specified in the "trial_webpage.html"
        file lines 28 and 89 for proper testing)
   NOTE: This process might take some time as it requires to load the BERT model first
c. Run the trial_webpage.html file on any browser you have (preferably Chrome) to test out the API calls
-------------------------------
