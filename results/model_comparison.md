# Base vs Fine-Tuned Model

- **Base model:** `HuggingFaceTB/SmolLM2-135M-Instruct`
- **Fine-tuned model:** `oscardean/smollm2-135m-text2cypher`
- **Evaluation split:** `test`

## Metric comparison

| Metric | Base | Fine-tuned | Delta |
|---|---:|---:|---:|
| `normalized_exact_match` | 0.00% | 0.00% | +0.00% |
| `canonicalized_exact_match` | 0.00% | 0.00% | +0.00% |
| `has_basic_query_structure` | 2.00% | 100.00% | +98.00% |
| `token_precision` | 8.74% | 62.54% | +53.80% |
| `token_recall` | 24.81% | 53.66% | +28.85% |
| `token_f1` | 12.35% | 55.20% | +42.85% |
| `node_labels_match` | 0.00% | 58.00% | +58.00% |
| `relationship_types_match` | 54.00% | 64.00% | +10.00% |
| `properties_match` | 16.00% | 44.00% | +28.00% |
| `directions_match` | 28.00% | 40.00% | +12.00% |
| `comparison_operators_match` | 48.00% | 42.00% | -6.00% |
| `component_match_rate` | 29.20% | 49.60% | +20.40% |

## Example comparisons

### Example 1

**Instance:** `instance_id_13430`

**Graph schema**

```text
Graph schema: Relevant node labels and their properties (with datatypes) are:
UpdateDate {update_date: DATE}
```

**Natural-language question**

Which nodes are connected to UpdateDate where update_date is 2008-01-29, and also to another node?

**Ground-truth Cypher**

```cypher
MATCH (a:UpdateDate{update_date:'2008-01-29'})-[r]->(n), (n)-[s]->(m) RETURN labels(n) AS Interim, labels(m) AS Target
```

**Base-model prediction**

```text
To answer this question, we need to traverse the graph and find the nodes that are connected to the update date. We start by traversing the nodes in the order of update_date. We can use the following steps: 1. Start with the node `UpdateDate` and its parent node `Node1`. 2. Move to the node `Node1` and its parent node `Node2`. 3. Move to the node `Node2` and its parent node `Node3`. 4. Move to the node `Node3` and its parent node `Node4`. 5. Move to the node `Node4` and its parent node `Node5`. 6. Move to the node `Node5` and its parent node `Node6`. 7. Move to the node `Node6` and its parent node `Node7`. 8. Move to the node `Node7` and its
```

**Fine-tuned-model prediction**

```cypher
MATCH (n:UpdateDate) WHERE n.update_date IS NOT NULL AND n.update_date IS NOT NULL AND n.update_date IS NOT NULL AND n.update_date IS NOT NULL AND n.update_date IS NOT NULL RETURN n
```

| Model | Exact match | Token F1 | Component match |
|---|---:|---:|---:|
| Base | 0.00% | 4.28% | 40.00% |
| Fine-tuned | 0.00% | 22.73% | 80.00% |

### Example 2

**Instance:** `instance_id_33953`

**Graph schema**

```text
Node properties:
- **Stream**
  - `createdAt`: DATE_TIME Min: 2007-05-22T10:39:53.259969Z, Max: 2021-05-09T14:16:40.370294Z
  - `id`: STRING Example: "129004176"
  - `description`: STRING Example: "Welcome to my Channel, I'm Big Chase, 31 living in"
  - `url`: STRING Example: "https://www.twitch.tv/itsbigchase"
  - `name`: STRING Example: "itsbigchase"
  - `followers`: INTEGER Min: 0, Max: 10243195
  - `total_view_count`: INTEGER Min: 21, Max: 1451487256
- **Game**
  - `name`: STRING Example: "Rust"
- **Language**
  - `name`: STRING Example: "en"
- **User**
  - `name`: STRING Example: "itsbigchase"
- **Team**
  - `createdAt`: DATE_TIME Min: 2011-10-11T23:59:43.548701Z, Max: 2021-05-07T17:19:09.946382Z
  - `name`: STRING Example: "ESO Stream Team"
  - `id`: STRING Example: "5893"
Relationship properties:

The relationships:
(:Stream)-[:PLAYS]->(:Game)
(:Stream)-[:HAS_LANGUAGE]->(:Language)
(:Stream)-[:MODERATOR]->(:Stream)
(:Stream)-[:MODERATOR]->(:User)
(:Stream)-[:CHATTER]->(:Stream)
(:Stream)-[:CHATTER]->(:User)
(:Stream)-[:HAS_TEAM]->(:Team)
(:Stream)-[:VIP]->(:Stream)
(:Stream)-[:VIP]->(:User)
(:User)-[:CHATTER]->(:Stream)
(:User)-[:CHATTER]->(:User)
(:User)-[:MODERATOR]->(:Stream)
(:User)-[:MODERATOR]->(:User)
```

**Natural-language question**

Find the streams that were created before 2018 and have more than 5,000 followers.

**Ground-truth Cypher**

```cypher
MATCH (s:Stream) WHERE s.createdAt < datetime('2018-01-01T00:00:00.000000Z') AND s.followers > 5000 RETURN s
```

**Base-model prediction**

```text
Node properties: - **Stream** - `createdAt`: DATE_TIME Min: 2011-10-11T23:59:43.548701Z, Max: 2021-05-07T17:19:09.946382Z - `id`: STRING Example: "5893" - `description`: STRING Example: "Welcome to my Channel, I'm Big Chase, 31 living in" - `url`: STRING Example: "https://www.twitch.tv/itsbigchase" - `name`: STRING Example: "itsbigchase" - `followers`: INTEGER Min: 0, Max: 10243195 - `total_view_count`:
```

**Fine-tuned-model prediction**

```cypher
MATCH (s:Stream)-[:PLAYS]->(s:Game) WHERE s.createdAt < (2018-01-01T00:00:00.000000Z)) RETURN s.name AS streamName, count(s) AS followers
```

| Model | Exact match | Token F1 | Component match |
|---|---:|---:|---:|
| Base | 0.00% | 7.62% | 40.00% |
| Fine-tuned | 0.00% | 50.00% | 0.00% |

### Example 3

**Instance:** `instance_id_26310`

**Graph schema**

```text
Node properties:
- **Question**
  - `favorites`: INTEGER Example: "0"
  - `answered`: BOOLEAN 
  - `text`: STRING Example: "### This is:  Bug    ### Specifications  OS: Win10"
  - `link`: STRING Example: "https://stackoverflow.com/questions/62224586/playg"
  - `createdAt`: DATE_TIME Min: 2020-06-05T16:57:19Z, Max: 2020-06-05T21:49:16Z
  - `title`: STRING Example: "Playground is not loading with apollo-server-lambd"
  - `id`: INTEGER Min: 62220505, Max: 62224586
  - `upVotes`: INTEGER Example: "0"
  - `score`: INTEGER Example: "-1"
  - `downVotes`: INTEGER Example: "1"
- **Tag**
  - `name`: STRING Example: "aws-lambda"
- **User**
  - `image`: STRING Example: "https://lh3.googleusercontent.com/-NcFYSuXU0nk/AAA"
  - `link`: STRING Example: "https://stackoverflow.com/users/10251021/alexandre"
  - `id`: INTEGER Min: 751, Max: 13681006
  - `reputation`: INTEGER Min: 1, Max: 420137
  - `display_name`: STRING Example: "Alexandre Le"
Relationship properties:

The relationships:
(:Question)-[:TAGGED]->(:Tag)
(:User)-[:ASKED]->(:Question)
```

**Natural-language question**

What are the top 5 tags used in questions with a score less than zero?

**Ground-truth Cypher**

```cypher
MATCH (q:Question)-[:TAGGED]->(t:Tag) WHERE q.score < 0 RETURN t.name, count(q) AS usage ORDER BY usage DESC LIMIT 5
```

**Base-model prediction**

```text
Based on the provided Cypher query, the following relationship properties are used: - `Question`: `(:Question)`: `(:Question)`: `(:Question)`: `(:Question)`: `(:Question)`: `(:Question)`: `(:Question)`: `(:Question)`: `(:Question)`: `(:Question)`: `(:Question)`: `(:Question)`: `(:Question)`: `(:Question)`: `(:Question)`: `(:Question)`: `(:Question)`: `(:Question)`: `(:Question)`: `(:Question)`: `(:Question)`: `(:Question)`: `(:Question)`: `(:Question)`: `(:Question)`: `(:Question)`: `(:Question)`: `(:Question)`: `(
```

**Fine-tuned-model prediction**

```cypher
MATCH (a:Answer) WHERE a.question_id <> 5 RETURN a.tag AS top_tags, a.score AS top_tags
```

| Model | Exact match | Token F1 | Component match |
|---|---:|---:|---:|
| Base | 0.00% | 12.12% | 0.00% |
| Fine-tuned | 0.00% | 37.50% | 0.00% |

### Example 4

**Instance:** `instance_id_30446`

**Graph schema**

```text
Node properties:
- **User**
  - `label`: STRING Example: "inchristbl.bsky.social"
  - `key`: STRING Example: "0"
  - `y`: FLOAT Example: "6000.659"
  - `color`: STRING Example: "#3CB371"
  - `x`: FLOAT Example: "-4383.217"
  - `area`: FLOAT Example: "2.6"
  - `size`: FLOAT Example: "1.82"
Relationship properties:
- **INTERACTED**
  - `weight: INTEGER` Example: "1"
  - `ogWeight: INTEGER` Example: "1"
  - `size: FLOAT` Example: "0.2"
  - `key: STRING` Example: "geid_68_0"
The relationships:
(:User)-[:INTERACTED]->(:User)
```

**Natural-language question**

List the labels of users with an area less than 2 and a size greater than 1.

**Ground-truth Cypher**

```cypher
MATCH (u:User) WHERE u.area < 2 AND u.size > 1 RETURN u.label
```

**Base-model prediction**

```text
The following Cypher queries are available for the given question: ``` SELECT * FROM users WHERE area < 2 AND size > 1; ``` The Cypher query returns the following results: ``` 0 ``` The Cypher query returns the following results: ``` 0 ``` The Cypher query returns the following results: ``` 0 ``` The Cypher query returns the following results: ``` 0 ``` The Cypher query returns the following results: ``` 0 ``` The Cypher query returns the following results: ``` 0 ``` The Cypher query returns the following results: ``` 0 ``` The Cypher query returns the following results: ``` 0 ``` The Cypher
```

**Fine-tuned-model prediction**

```cypher
MATCH (u:User)-[:INTERACTED]->(m:FLOAT) WHERE m.area < 2 AND m.size > 1 RETURN labels(m)
```

| Model | Exact match | Token F1 | Component match |
|---|---:|---:|---:|
| Base | 0.00% | 15.00% | 60.00% |
| Fine-tuned | 0.00% | 60.71% | 20.00% |

### Example 5

**Instance:** `instance_id_19443`

**Graph schema**

```text
Graph schema: Relevant node labels and their properties (with datatypes) are:
Article {title: STRING}
Article {comments: STRING}
```

**Natural-language question**

Find the title and the comments for those Article where title starts with O!

**Ground-truth Cypher**

```cypher
MATCH (n:Article) WHERE n.title STARTS WITH 'O' RETURN n.title AS title, n.comments AS comments
```

**Base-model prediction**

```text
c // Find the title and comments for Article where title starts with O! const title = Article.title.toLowerCase(); const comments = Article.comments.toLowerCase(); // Return the title and comments for Article where title starts with O! const titleQuery = `Article {title: ${title}}, Comments {comments: ${comments}}`; // Return the title and comments for Article where title starts with O! const titleQuery = `Article {title: ${title}}, Comments {comments: ${comments}}`; // Return the title and comments for Article where title starts with O! const titleQuery = `Article {title: ${title}}, Comments {comments: ${comments}}`; // Return the title and comments for Article where title starts with O! const titleQuery = `Article {title: ${
```

**Fine-tuned-model prediction**

```cypher
MATCH (n:Article) WHERE n.title LIKE '%O%' RETURN n.title AS title, n.comments AS comments
```

| Model | Exact match | Token F1 | Component match |
|---|---:|---:|---:|
| Base | 0.00% | 13.76% | 60.00% |
| Fine-tuned | 0.00% | 89.80% | 100.00% |
