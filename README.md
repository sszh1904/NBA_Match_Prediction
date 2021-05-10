# NBA-Match-Predictor

Automated Daily Routine @12am US timing:
1. Get game results for previous day
2. For each game:
a. Extract prediction
b. Combine results with prediction (df)
c. Update team stats
3. Append df into "games history"
4. Reset "upcoming games"
5. Get matchups for today
6. For each matchup:
a. Get both team's stats
b. Make prediction
c. Add to upcoming games list