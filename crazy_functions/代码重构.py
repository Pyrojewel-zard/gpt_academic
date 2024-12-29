from loguru import logger
from toolbox import update_ui
from toolbox import CatchException, report_exception
from toolbox import write_history_to_file, promote_file_to_downloadzone
from crazy_functions.crazy_utils import request_gpt_model_in_new_thread_with_ui_alive
import time, os, glob

def 重构代码(file_manifest, project_folder, llm_kwargs, plugin_kwargs, chatbot, history, system_prompt):
    logger.info('begin analysis on:', file_manifest)
    for index, fp in enumerate(file_manifest):
        with open(fp, 'r', encoding='utf-8', errors='replace') as f:
            file_content = f.read()

        i_say = (f'请对下面的程序文件进行重构，优化代码结构和性能，可以采取通过将重复的代码块封装成函数调用的方式来优化代码等方式，并附上重构后的代码，'
                 f'使用diff格式展示更改，文件名是{os.path.relpath(fp, project_folder)}，'
                 f'文件内容是 ```{file_content}```')
        i_say_show_user = f'[{index+1}/{len(file_manifest)}] 请对下面的程序文件进行重构并优化: {os.path.abspath(fp)}'
        chatbot.append((i_say_show_user, "[Local Message] waiting gpt response."))
        yield from update_ui(chatbot=chatbot, history=history)  # Refresh the interface

        msg = '正常'
        # ** gpt request **
        gpt_say = yield from request_gpt_model_in_new_thread_with_ui_alive(
            i_say, i_say_show_user, llm_kwargs, chatbot, history=[], sys_prompt=system_prompt)  # With timeout countdown

        chatbot[-1] = (i_say_show_user, gpt_say)
        history.append(i_say_show_user); history.append(gpt_say)
        yield from update_ui(chatbot=chatbot, history=history, msg=msg)  # Refresh the interface
        time.sleep(2)

    res = write_history_to_file(history)
    promote_file_to_downloadzone(res, chatbot=chatbot)
    chatbot.append(("完成了吗？", res))
    yield from update_ui(chatbot=chatbot, history=history, msg=msg)  # Refresh the interface

@CatchException
# 主流代码的后缀有：.py, .cpp, .c, .h, .hpp, .css,.html,.js,.java,.cs,.php,.go,.rs,.rb,.sh,.pl,.lua,.swift,.kt,.m,.mm,.ts,.dart,.r,.jl,.v,.vhdl,.vhd,.sv,.svh,.sby,.smt2,.z3,.csp,.mzn,.fzn,.dzn,.lp,.mps,.mpl,.mod,.sm,.sp,.pyt,.pyw,.pyi,.pyx,.pxd,.pxi,.rpy,.rpyc,il
def 批量重构代码(txt, llm_kwargs, plugin_kwargs, chatbot, history, system_prompt, user_request, extensions=['.py', '.cpp', '.c', '.h', '.hpp', '.css', '.html', '.js', '.java', '.cs', '.php', '.go', '.rs', '.rb', '.sh', '.pl', '.lua', '.swift', '.kt', '.m', '.mm', '.ts', '.dart', '.r', '.jl', '.v', '.vhdl', '.vhd', '.sv', '.svh', '.sby', '.smt2', '.z3', '.csp', '.mzn', '.fzn', '.dzn', '.lp', '.mps', '.mpl', '.mod', '.sm', '.sp', '.pyt', '.pyw', '.pyi', '.pyx', '.pxd', '.pxi', '.rpy', '.rpyc', 'il']):
    history = []    # Clear history to avoid overflow
    if os.path.exists(txt):
        project_folder = txt
    else:
        if txt == "":
            txt = '空空如也的输入栏'
        report_exception(chatbot, history, a=f"解析项目: {txt}", b=f"找不到本地项目或无权访问: {txt}")
        yield from update_ui(chatbot=chatbot, history=history)  # Refresh the interface
        return

    # Create a file manifest based on the provided extensions
    file_manifest = []
    for ext in extensions:
        file_manifest.extend(glob.glob(f'{project_folder}/**/*{ext}', recursive=True))

    if len(file_manifest) == 0:
        report_exception(chatbot, history, a=f"解析项目: {txt}", b=f"找不到任何文件: {extensions} 在: {txt}")
        yield from update_ui(chatbot=chatbot, history=history)  # Refresh the interface
        return

    yield from 重构代码(file_manifest, project_folder, llm_kwargs, plugin_kwargs, chatbot, history, system_prompt)