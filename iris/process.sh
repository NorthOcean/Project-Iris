###
 # @Author: Conghao Wong
 # @Date: 2021-05-26 15:07:51
 # @LastEditors: Conghao Wong
 # @LastEditTime: 2021-05-26 16:12:31
 # @Description: file content
 # @Github: https://github.com/conghaowoooong
 # Copyright 2021 Conghao Wong, All Rights Reserved.
###

FILE=$1

WITHP='(\\[A-Z]+)(~)'
WITHR='\1 '

WITHOUTP='(\\[A-Z]+)( )'
WITHOUTR='\1~'

if [ `grep -c '[A-Z]*[A-Z]~' $FILE` -ne '0' ];then
    sed -i '' -E "s/$WITHP/$WITHR/g" $FILE
    echo "~Exist! Change it!"
else
    sed -i "" -E "s/$WITHOUTP/$WITHOUTR/g" $FILE
    echo "~Do not exist! Add it!"
fi
