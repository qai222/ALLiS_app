Overview
---
This is a visualization app for the **A**ctive **L**earning **Li**gand **S**election project.

The goal of this project is to identify ligands that can enhance emission efficiency
and stability of perovskite nanocrystals using active learning suggestions.

deploy
---
1. dump svg files to the [assets/svg](assets/svg) folder
2. create a `mongo.env` file defines environmental variables
   ```
    ALLIS_MONGO_USR=<usr>
    ALLIS_MONGO_PWD=<pwd>
   ```
3. run docker compose
    ```
   docker compose --env-file mongo.env up -d
   ```