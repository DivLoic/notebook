#!/usr/bin/python
# coding: utf-8

"""
simple function to create a custum git-hook.
Author : Loic M. Divad
Date : 19 Apr 2017
please see https://github.com/DivLoic/notebook/blob/master/git-hooks/hook-commit.ipynb
"""

import sys, os, re
import fileinput
import subprocess
from urllib import quote_plus

e1 = u'\U0001F37A'
e2 = u'\U0001F37B'
e3 = u'\U0001F389'
BRANCH_PATTERN=r"branch[=,/]([\w,%,-]*)[/,)]"

def applyBranch(line, branch_name):
    branch = extractBranch(line)
    return line.replace(branch, quote_plus(branch_name)) if branch else line

def extractBranch(line):
    match = re.search(BRANCH_PATTERN, line)
    return match.group(1) if match else None

def editRmd(file_name, branch_name, func):
    stream = fileinput.FileInput(file_name, inplace=True)
    for line in stream:
        print func(line, branch_name),

    stream.close()


if __name__ == '__main__':

    BRANCH = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).strip()

    editRmd("README.md", BRANCH, applyBranch)

    subprocess.check_output(["git", "add", "README.md"])

    print u'{}{}{} git-hook (pre-commit) correctly apply'.format(e1, e2, e3)