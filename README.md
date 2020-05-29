# Modified Fuzzy Delphi Method with Optimized Consistency Aggregation Method

## Consistency Aggregation Method

Implementation of Modified Fuzzy Delphi Method with Optimized Consistency Aggregation Method (CAM). This script can be used to aggregate experts opinions on ranking of defined items based on conducted questionary survey. Survey is expected to be a list of questions with next answer options: 
* 'Strongly agree'
* 'Agree'
* 'Neutral'
* 'Disagree'
* 'Strongly disagree'

Additionally, each expert should provide information for researcher to define empirical "expert importance" coefficient that would correspond to expert's knowledge and experience in particular research area.  
  
Script generates two numeric values and single boolean value for each item. Next values are generated:
* *'Name'* is item name;
* *'Rank'* is obtained defuzzified rank of question;
* *'Consensus'* is obtained conensus rate;
* *'Verdict'* is verdict ("Retained" / "Discarder") based on `S` threshold value.

Script for CAM algorithm can be found in `cam.py`.   

## Statistical testing to ensure convergence

There is second script presented which helps evaluate stability and convergence of obtained group opinions and consensus rates for each item.  
  
Next statistical tests are used:
* Mann-Whitney test;
* Median test;
* Kruskal-Wallis test;

*p-value* of *0.05* is used as a threshold for accepting/rejecting null-hypothesis in all tests.
  
Corresponding script can be found in `stats.py`.   

# License
MIT

# Authors
[Ihor Markevych](mailto:ih.markevych@gmail.com) and [Egbe-Etu Emmanuel Etu](mailto:fw7443@wayne.edu)
