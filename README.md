# Rocket League Skills Detection 

This repo holds the code for the paper submited to IEEE Conference on Games 2020 "A Behavioral Pattern Mining Approach to Model Player Skills in Rocket League".

## Same skill, different sequences
First we want to illustrate that the sequence of inputs of the player will vary a lot when performing the same skill.
A first example of ceiling shot:
### Ceiling shot 1
<a href="https://www.youtube.com/watch?v=ybQJ1hs1slE
" target="_blank"><img src="https://raw.githubusercontent.com/Romathonat/RocketLeagueSkillsDetection/master/images/ceiling_shot_1_mini.png" border="10" /></a>

As inputs of the controller create a long sequence, we propose a visual representation of it. The following schema reprents buttons pressed over time.

![](https://raw.githubusercontent.com/Romathonat/RocketLeagueSkillsDetection/master/images/ceiling_shot_1.png)

Now let's look at a second example
### Ceiling shot 2
<a href="https://www.youtube.com/watch?v=WlWMyznvTj4" target="_blank"><img src="https://raw.githubusercontent.com/Romathonat/RocketLeagueSkillsDetection/master/images/ceiling_shot_2_mini.png" border="10" /></a>

![](https://raw.githubusercontent.com/Romathonat/RocketLeagueSkillsDetection/master/images/ceiling_shot_2.png)

Clearly, sequences of inputs created are different. However, there are (more or less) hidden patterns in those figures. The aim of this work is to automatically discover them, so that we will be able to classify figures in real time. This project is a proof-of-concept, but we showed in the paper that this system could integrated in-game, to classify players actions.

## List of skills
Here we list an example of each skill we want to be able to detect in the scope of this project.

### Powershot
<a href="https://www.youtube.com/watch?v=7D_QwT7jJxg" target="_blank"><img src="https://raw.githubusercontent.com/Romathonat/RocketLeagueSkillsDetection/master/images/powershot_mini.png" border="10" /></a>

![](https://raw.githubusercontent.com/Romathonat/RocketLeagueSkillsDetection/master/images/powershot.png)

### Waving Dash
<a href="https://www.youtube.com/watch?v=-eqBV1e0VVc" target="_blank"><img src="https://raw.githubusercontent.com/Romathonat/RocketLeagueSkillsDetection/master/images/waving_dash_mini.png" border="10" /></a>

![](https://raw.githubusercontent.com/Romathonat/RocketLeagueSkillsDetection/master/images/waving_dash.png)

### Air dribble
<a href="https://www.youtube.com/watch?v=EtEqgPVkC1U" target="_blank"><img src="https://raw.githubusercontent.com/Romathonat/RocketLeagueSkillsDetection/master/images/air_dribble_mini.png" border="10" /></a>

![](https://raw.githubusercontent.com/Romathonat/RocketLeagueSkillsDetection/master/images/air_dribble.png)

### Front Flick
<a href="https://www.youtube.com/watch?v=ArYibJO4sK8" target="_blank"><img src="https://raw.githubusercontent.com/Romathonat/RocketLeagueSkillsDetection/master/images/front_flick_mini.png" border="10" /></a>

![](https://raw.githubusercontent.com/Romathonat/RocketLeagueSkillsDetection/master/images/front_flick.png)

### Musty Flick
<a href="https://www.youtube.com/watch?v=o0FwER0dFgE" target="_blank"><img src="https://raw.githubusercontent.com/Romathonat/RocketLeagueSkillsDetection/master/images/musty_flick_mini.png" border="10" /></a>

![](https://raw.githubusercontent.com/Romathonat/RocketLeagueSkillsDetection/master/images/musyt_flick.png)
