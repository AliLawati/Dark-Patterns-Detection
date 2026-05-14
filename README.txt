----------------------------------------------------
Welcome to the Dark Patterns Classification Project!
----------------------------------------------------
NOTE: All of the code in this project was written and run using PyCharm IDE.

Please follow the instructions written below before running any file in the project to make sure that everything works well.

1. Python and PyCharm Installation:
-----------------------------------
a. Install Python version 3.10.0 from https://www.python.org/downloads/release/python-3100/
b. Run the installer and follow the instructions
c. Install PyCharm Community Edition from https://www.jetbrains.com/pycharm/download
d. Run the installer and follow the instructions
e. Run PyCharm, open the project, and wait for it to load.
f. After the project is ready, click in the bottom right corner (where written <No Interpreter>)
     i. Select "Add New Interpreter"
    ii. Select "Add Local Interpreter"
   iii. Add your Python 3.10 interpreter as "Virtualenv"
-----------------------------------

2. Required Dependencies Installation:
--------------------------------------
a. After the interpreter is ready, install the requirements using the cli in PyCharm IDE bottom left icon (terminal)
b. Type the command: pip install -r requirements.txt
--------------------------------------

3. CUDA Toolkit and cuDNN Installation (if you have a GPU for NVIDIA):
----------------------------------------------------------------------
a. Install CUDA Toolkit 11.2 for the appropriate operating system from https://developer.nvidia.com/cuda-11.2.0-download-archive
b. Run the installer and follow the instructions
c. Download cuDNN v8.1.0 (January 26th, 2021), for CUDA 11.0,11.1 and 11.2 for the appropriate operating system from https://developer.nvidia.com/rdp/cudnn-archive (login to your NVIDIA account if required, or sign up if you don't already have an account)
d. Extract the downloaded .zip file
e. Copy the files in the "bin" folder of the extracted file, and paste them in the "bin" folder of the CUDA Toolkit (usually found in "C:\Program Files\NVIDIA GPU Computing Toolkit")
f. Repeat step 'e' for the files found in the "include" and "lib/x64" folders of the extracted file (paste them in the "include" and "lib/x64" folders of the CUDA Toolkit respectively)
g. Make sure that the CUDA Toolkit has been added in the system/user variables of your device. If not, follow the following instructions:
     i. Search for the environment variables on your device.
     ii. Under system variables, add the following variables along with their values:
          ● "CUDA_PATH": C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2 (or wherever the CUDA Toolkit is installed on your device)
          ● "CUDA_PATH_V11_2": C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2 (or wherever the CUDA Toolkit is installed on your device)
          ● "NVCUDASAMPLES_ROOT": C:\ProgramData\NVIDIA Corporation\CUDA Samples\v11.2 (or wherever the CUDA Toolkit is installed on your device)
          ● "NVCUDASAMPLES11_2_ROOT": C:\ProgramData\NVIDIA Corporation\CUDA Samples\v11.2 (or wherever the CUDA Toolkit is installed on your device)
     iii. Click on the "Path" variable in the system variables, and click on the "Edit..." button
     iv. Make sure that the following 2 paths are added in it. If not, add them:
          ● C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin (or wherever the CUDA Toolkit is installed on your device)
          ● C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\libnvvp (or wherever the CUDA Toolkit is installed on your device)
     v. Save and apply all the changes, and restart your device.
----------------------------------------------------------------------

4. Model Installation:
----------------------
a. Install ollama for the appropriate operating system from https://ollama.com/
b. Run the installer and follow the instructions
c. Run this command in cli: ollama run mistral
d. After the model is installed in the system, you can exit the cli by typing: /bye OR /exit
----------------------

5. Running and Testing the API:
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
