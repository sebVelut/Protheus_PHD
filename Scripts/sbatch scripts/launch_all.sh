#!/bin/bash

# first, we submit the original job, and capture the output. 
# Slurm prints "submitted batch job <jobid>" when we submit a job.
# We store that output in jobstring using $( ... )

jobstring1=$(sbatch scripts_sub1.sh)

# The last word in jobstring is the job ID. There are several ways to get it,
# but the shortest is with parameter expansion: ${jobstring##* }

jobid1=${jobstring1##* }

# Now submit j2.slurm as a dependant job to j1:

jobstring2=$(sbatch --dependency=afterany:${jobid1} scripts_sub2.sh)
jobid2=${jobstring2##* }

# Now submit j3.slurm as a dependant job to j2:

jobstring3=$(sbatch --dependency=afterany:${jobid2} scripts_sub3.sh)
jobid3=${jobstring3##* }

# Now submit j4.slurm as a dependant job to j3:

jobstring4=$(sbatch --dependency=afterany:${jobid3} scripts_sub4.sh)
jobid4=${jobstring4##* }

# Now submit j5.slurm as a dependant job to j4:

jobstring5=$(sbatch --dependency=afterany:${jobid4} scripts_sub5.sh)
jobid5=${jobstring5##* }

# Now submit j6.slurm as a dependant job to j5:

jobstring6=$(sbatch --dependency=afterany:${jobid5} scripts_sub6.sh)
jobid6=${jobstring6##* }

# Now parralel work with j7:

jobstring7=$(sbatch scripts_sub7.sh)
jobid7=${jobstring7##* }

# Now submit j8.slurm as a dependant job to j7:

jobstring8=$(sbatch --dependency=afterany:${jobid7} scripts_sub8.sh)
jobid8=${jobstring8##* }

# Now submit j9.slurm as a dependant job to j8:

jobstring9=$(sbatch --dependency=afterany:${jobid8} scripts_sub9.sh)
jobid9=${jobstring9##* }

# Now submit j10.slurm as a dependant job to j9:

jobstring10=$(sbatch --dependency=afterany:${jobid9} scripts_sub10.sh)
jobid10=${jobstring10##* }

# Now submit j11.slurm as a dependant job to j10:

jobstring11=$(sbatch --dependency=afterany:${jobid10} scripts_sub11.sh)
jobid11=${jobstring11##* }

# Now submit j12.slurm as a dependant job to j11:

jobstring12=$(sbatch --dependency=afterany:${jobid11} scripts_sub12.sh)
jobid12=${jobstring12##* }