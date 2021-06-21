'''
Author: Conghao Wong
Date: 2021-04-05 19:01:15
LastEditors: Conghao Wong
LastEditTime: 2021-04-06 11:24:05
Description: file content
'''

import re

FILE_NAME = './modules/models/_prediction/_utils.py'
SAVE_PATH = './clog.txt'

def read_com(file_path, save_path, max_search_line=50):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    all_results = []
    for index, line in enumerate(lines):
        if 'def' in line.split(' '):
            function_name = re.findall('def .*\(', line)[0][4:-1]
            if '"""\n' in lines[index+1].split(' '):
                start_line = index + 2
                jump = False
            else:
                continue
            
            length = 1
            while not '"""\n' in lines[index+1+length].split(' '):
                length += 1
                if length >= max_search_line:
                    jump = True
                    break
            
            if not jump:
                end_line = index + length
                current_lines = lines[start_line:end_line]
            
            summary_line_index = []
            useful_lines = []
            for l_index, c_line in enumerate(current_lines):
                if re.sub(' *', '', c_line) == '\n':
                    continue
                
                for flag in ['param', 'returns', 'return']:
                    if flag in c_line:
                        current_lines[l_index] = re.sub('    ', '', current_lines[l_index])
                        current_lines[l_index] = re.sub(':{} '.format(flag), '<{} name="'.format(flag), current_lines[l_index])
                        current_lines[l_index] = re.sub(': +', '"> '.format(flag), current_lines[l_index])
                        current_lines[l_index] = re.sub('\n', ' </{}>\n'.format(flag), current_lines[l_index])
                        useful_lines.append(l_index)
                
            
            results = (
                ['///' + 'FUNCTION_NAME: {}\n'.format(function_name)]
                + ['///' + '<summary>\n'] 
                + [('///' + current_lines[i]) for i in range(len(current_lines)) if not i in useful_lines] 
                + ['///' + '</summary>\n'] 
                + [('///' + current_lines[i]) for i in useful_lines]
                + ['\n\n']
            )
            all_results += results
        
    with open(save_path, 'w+') as f:
        f.writelines(all_results)
                

if __name__ == '__main__':
    read_com(FILE_NAME, SAVE_PATH)