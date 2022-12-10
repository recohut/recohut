# Recohut Data Bootcamps

Recohut Provides the following bootcamps:

1. Data Engineering
2. Data Science

We follow 2 simple principle for outstanding results:

1. Agile
2. Workshops

##  Workshop Process

```mermaid
flowchart TD
    A[Start] --> C[Understand the Workshop overview and Prerequisites]
    subgraph Prerequisite
    C --> D[Complete the Prerequisites]
    D --> E{Workshop Prerequisite Test}
    end
    subgraph Development
    E --> |Passed| B[Setup the initial workshop template in your git]
    E --> |Failed| C
    B --> F[Develop the Workshop - Test driven development]
    F --> G[Request for the Peer review and get the sign-off]
    G --> H{Signed-off}
    end
    subgraph Production
    H --> |No| F
    H --> |Yes| I[Deploy the Workshop]
    I --> J[Complete the Documentation]
    J --> K[Release the Workshop]
    K --> L[Add the workshop in your resume - Resume Buildup]
    L --> M[Take the Workshop Mock Interview - Full Mock]
    end
    M --> N[End]
    click C callback "Tooltip for a callback"
```

## Prerequisites

1. Python & SQL - You should have at least intermediate-level knowledge of Python and SQL.
2. Time - You need to commit at least 4 hours per day for a time span of 8-10 weeks to gain the skills.
