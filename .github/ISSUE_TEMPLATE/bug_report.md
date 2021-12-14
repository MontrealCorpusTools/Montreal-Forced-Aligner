---
name: Bug report
about: Create a report to help us improve
title: "[BUG]"
labels: bug
assignees: mmcauliffe

---

**Debugging checklist**

[ ] Have you updated to latest MFA version?
[ ] Have you tried rerunning the command with the `--clean` flag?

**Describe the issue**
A clear and concise description of what the bug is.

**For Reproducing your issue**
Please fill out the following:

1. Corpus structure
   * What language is the corpus in?
   * How many files/speakers?
   * Are you using lab files or TextGrid files for input?
2. Dictionary
   * Are you using a dictionary from MFA? If so, which one?
   * If it's a custom dictionary, what is the phoneset?
3. Acoustic model
   * If you're using an acoustic model, is it one download through MFA? If so, which one?
   * If it's a model you've trained, what data was it trained on?

**Log file**
Please attach the log file for the run that encountered an error (by default these will be stored in `~/Documents/MFA`).

**Desktop (please complete the following information):**
 - OS: [e.g. Windows, OSX, Linux]
 - Version [e.g. MacOSX 10.15, Ubuntu 20.04, Windows 10, etc]
 - Any other details about the setup (Cloud, Docker, etc)

**Additional context**
Add any other context about the problem here.
