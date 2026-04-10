"""
Used for testing
"""


from pathlib import Path

pairs = ["0", "1"]
splits = ["550_59"]

def get_num_temp(group):
    if group == "1_0":
        return "1"
    elif group == "4_1":
        return "4"
    else:
        return "8"
    
template = """
domain="{domain}"
split="{split}"
num_temp="{num_temp}"
gender="{gender}"
run="{run}"
num_templates="{num_temp}"

python test_dp.py \
    --save_name ${{split}}_${{gender}}_${{run}}_test \
    --save_dir ./models/${{split}}_${{gender}}_${{run}}_test \
    --cfg  \
    --csv_train  \
    --csv_infer  \
    --imgs_train  \ 
    --imgs_infer  \    
    --txt_train  \
    --txt_infer  \ 
    --weights_global  \
    --weights_local  \
    --num_templates ${{num_templates}} \
    --meta 
"""

for split in splits:
    for domain in ["hand"]:
        for run in pairs:
            for gender in ["M", "F", "C", "OM", "OF", "YM", "YF", "CC"]:
                jobname = f"{split}_{gender}_{run}_test"
                out = f"{jobname}.out"
                err = f"{jobname}.err"
                filename = Path(f"{jobname}.sh") 
                content = template.format(jobname=jobname, out=out, err=err, split=split, domain=domain, num_temp=get_num_temp(split), type=type, gender=gender, run = run)
                filename.write_text(content)
                filename.chmod(0o755)
                print("Wrote", filename)

print("")

