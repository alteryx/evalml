reqs_list=$(python -c "import os; import sys; import io; save_stdout = sys.stdout; sys.stdout = open('trash', 'w'); from evalml.utils import get_evalml_pip_requirements; sys.stdout = save_stdout; print('\n'.join(get_evalml_pip_requirements(os.getcwd())));")
allow_list=$(echo ${reqs_list} | grep -oE "[a-zA-Z]+[a-zA-Z_\-]*" | paste -d "|" -s -)
echo "Allow list: ${allow_list}"
pip freeze | grep -v "evalml.git" | grep -E "${allow_list}" > "${DEPENDENCY_FILE_PATH}"
