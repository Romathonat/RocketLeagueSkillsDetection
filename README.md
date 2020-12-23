# Rocket League Skills Detection 

This repo holds the code for the paper submited to IEEE Conference on Games 2020 "A Behavioral Pattern Mining Approach to Model Player Skills in Rocket League" (you can access it freely [here](https://www.researchgate.net/publication/343852416_A_Behavioral_Pattern_Mining_Approach_to_Model_Player_Skills_in_Rocket_League)). This is an application paper that uses the algorithm SeqScout from [1]. You can also see a video presentation of the project [here](https://www.youtube.com/watch?v=0zUlOIaDzqs&feature=youtu.be&ab_channel=IEEECOG)

## Same skill, different sequences
First we want to illustrate that the sequence of inputs of the player will vary a lot when performing the same skill.
The first example is a ceiling shot:
### Ceiling shot 1
<a href="https://www.youtube.com/watch?v=ybQJ1hs1slE
" target="_blank"><img src="https://raw.githubusercontent.com/Romathonat/RocketLeagueSkillsDetection/master/images/ceiling_shot_1_mini.png" border="10" /></a>

As inputs of the controller create a long sequence, we propose a visual representation of it. The following schema presents buttons pressed over time.

![](https://raw.githubusercontent.com/Romathonat/RocketLeagueSkillsDetection/master/images/ceiling_shot_1.png)

Here is the legend:
<img src="https://raw.githubusercontent.com/Romathonat/RocketLeagueSkillsDetection/master/images/legend.png" alt="drawing" width="120"/>


Now let's look at a second example of ceiling shot
### Ceiling shot 2
<a href="https://www.youtube.com/watch?v=WlWMyznvTj4" target="_blank"><img src="https://raw.githubusercontent.com/Romathonat/RocketLeagueSkillsDetection/master/images/ceiling_shot_2_mini.png" border="10" /></a>

![](https://raw.githubusercontent.com/Romathonat/RocketLeagueSkillsDetection/master/images/ceiling_shot_2.png)

Clearly, sequences of inputs created are different, even if they are different instances of **the same figure**. However, they contains (more or less) hidden patterns:


![](https://raw.githubusercontent.com/Romathonat/RocketLeagueSkillsDetection/master/images/ceiling_2_pattern.png) 
![](https://raw.githubusercontent.com/Romathonat/RocketLeagueSkillsDetection/master/images/ceiling_1_pattern.png)


**The aim of this work is to automatically discover them, so that we will be able to classify figures in real time**. 
  
**Important**: Note that here we only show sequences of inputs for simplification. However, there is a lot of game information we used in our workflow that is not represented here, like positions, speed, distance to the ball etc (see paper).

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

## Conclusion
This project is a proof-of-concept, but we showed in the paper that this system could be integrated in-game, in order to classify players actions. This could help improving a better ranking system for players, or to create new game modes based on skills performances (like the [horse](https://fr.wikipedia.org/wiki/HORSE_(basket-ball)) game). Note also that it could be interesting to create a mode where new players could learn special skills in a "[Guitar Hero](https://en.wikipedia.org/wiki/Guitar_Hero) fashion": for a specific shot in training, inputs would be displayed, and player would have to press them correctly to succesfuly perform the shot.

[1] R. Mathonat, D. Nurbakova, J. Boulicaut and M. Kaytoue, "SeqScout: Using a Bandit Model to Discover Interesting Subgroups in Labeled Sequences," 2019 IEEE International Conference on Data Science and Advanced Analytics (DSAA), Washington, DC, USA, 2019, pp. 81-90.
