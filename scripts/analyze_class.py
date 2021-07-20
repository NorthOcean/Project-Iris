"""
@Author: Conghao Wong
@Date: 2021-04-16 16:02:45
@LastEditors: Conghao Wong
@LastEditTime: 2021-07-20 11:11:29
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

import inspect
import sys
import os

sys.path.insert(0, os.path.abspath('.'))
import modules as mod


def get_menmber(package):
    return [item for _, item in inspect.getmembers(package) if (inspect.isclass(item) or inspect.ismodule(item)) and (not item.__name__.startswith('_'))]


def print_all_members(package):
    waiting = get_menmber(package)
    name_list = []
    info_list = []
    lens = ['```mermaid\n',
            '    graph LR\n']

    while len(waiting):
        item = waiting.pop(-1)
        if inspect.ismodule(item):
            if item.__name__.split('.')[0] in ['modules']:
                waiting += get_menmber(item)
        else:
            father_name = item.__base__.__name__
            father_module = item.__base__.__module__
            name = item.__name__
            module = item.__module__

            if name != 'builtins':
                class_name = '{}/{}'.format(module, name)
                info = '{}_{}'.format(module, name)
                father_info = '{}_{}'.format(father_module, father_name)

                if not class_name in name_list:
                    print(class_name)
                    print(father_info)
                    lens.append('        {}("{}({})") --> {}("{}({})")\n'.format(
                        father_info, father_name, father_module, info, name, module))
                    name_list.append(class_name)
                    info_list.append(father_info)

                    if father_name != 'object':
                        waiting += get_menmber(item)

    lens.append('```\n')
    return lens


def update_readme_file(file_path, new_lines, start: str, end: str):
    with open(file_path, 'r') as f:
        all_lines = f.readlines()

    start_line = -1
    end_line = -1
    for index, line in enumerate(all_lines):
        if line.startswith(start):
            start_line = index

        if line.startswith(end) and start_line > 0:
            end_line = index
            break

    if start_line > 0 and end_line > 0:
        write_lines = all_lines[:start_line+1] + \
            new_lines + all_lines[end_line:]
        with open(file_path, 'w+') as f:
            f.writelines(write_lines)

        print('File update success.')


lines = print_all_members(mod)
update_readme_file('./classRef.md', lines,
                   start='<!-- GRAPH BEGINS HERE -->',
                   end='<!-- GRAPH ENDS HERE -->')
