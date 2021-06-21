###
 # @Author: Conghao Wong
 # @Date: 2021-03-22 14:40:45
 # @LastEditors: Conghao Wong
 # @LastEditTime: 2021-04-08 14:41:22
 # @Description: file content
### 

# satoshi alpha
# for dataset in zara2 univ
# do
#     for K in 5 8 10 15 20 30
#     do
#         python main.py --load ./newSatoshiLogs/sa_K${K}_${dataset} --draw_results 0
#     done
# done

# satoshi beta
# for dataset in eth hotel zara1 zara2 univ
# do
#     for h in 1 3 5 8
#     do
#         python main.py --load ./newSatoshiLogs/sb_H${h}_${dataset}
#     done
# done

# full satoshi
# for dataset in eth hotel zara1 zara2 univ
# do
#     for K in 5 8 10 15 20 30
#     do
#         for h in 0
#         do
#             python main.py \
#                 --model test \
#                 --loada ./newSatoshiLogs/sa_K${K}_${dataset} \
#                 --loadb l./newSatoshiLogs/sb_H${h}_${dataset}
#         done
#     done
# done

# satoshi alpha 8 -> 20
for dataset in eth hotel zara1 zara2
do
    python main.py --load ./8to20logs/8to20sa${dataset} --draw_results 1
done