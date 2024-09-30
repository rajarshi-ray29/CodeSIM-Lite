docker run -v $(pwd):/app ganler/evalplus:latest --dataset humaneval --samples C:/Users/CSE2/Desktop/SCoder/results/HumanEvalPlus/Direct/GPT4/Python3-0.2-0.1-1/Run-1/Results.jsonl




cp {EP_RESULTS_PATH} results/GPT4-Direct-Results.jsonl

docker run -v $(pwd):/app ganler/evalplus:latest --dataset humaneval --samples /app/results/GPT4-Direct-Results.jsonl > results/summary.txt

cp results/GPT4-Direct-Results_eval_results.json {EP_RESULTS_PATH}

mv results/summary.txt  /mnt/c/Users/CSE2/Desktop/SCoder/results/HumanEvalPlus/SCoder/GPT4/Python3-0.2-0.1-1/Run-1/

rm -rf results/*

EvalResults.json