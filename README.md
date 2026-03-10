# American Inequality Mirror Counties

**A tool for identifying demographically similar US counties to support policy research, advocacy, and community analysis.**

🔗 [Live App](https://americaninequalityproject.streamlit.app/) | 📰 [American Inequality on Substack](https://americaninequality.substack.com/?utm_campaign=profile_chips)

---

## What This Is

Mirror Counties is an interactive data tool that finds US counties that are demographically and economically similar to any selected county. The goal is to help policymakers, researchers, and journalists identify communities that share comparable characteristics — and use those comparisons to surface learning opportunities, inform local policy, and understand inequality at a granular level.

If a county is struggling with low income, poor health outcomes, or limited upward mobility, finding a demographically similar county that is performing better can reveal concrete, transferable policy lessons. If a county has made meaningful progress, understanding its mirrors helps identify where those gains could be replicated.

The tool is built and maintained by [Michael Tiede](https://substack.com/@michaeltiede/posts) as part of the American Inequality Project, which publishes data-driven research on economic inequality, healthcare access, and social disparities in the United States.

---

## Features

- **Mirror County Matching** — Select any US county and find its closest demographic and economic matches nationwide
- **Comparison Charts** — Visualize how your selected county compares to its mirrors on income, life expectancy, and upward mobility
- **Interactive Map** — See where mirror counties are located geographically
- **Culture Analysis** — AI-powered qualitative analysis comparing the selected county and its mirrors across historical, cultural, and economic dimensions
- **Manual Controls** — Advanced users can override algorithmic assumptions and adjust population filters and feature weights manually

---

## How It Works

### Data Sources

The app uses a combination of US Census / American Community Survey (ACS) data and custom-compiled demographic and economic data including:

- **Demographic composition** — racial and ethnic population percentages at the county level
- **Economic indicators** — median household income, upward mobility scores
- **Health outcomes** — life expectancy
- **Industry composition** — primary and secondary industries by county
- **Geography** — rural/urban classification, latitude/longitude, FIPS codes

### Matching Algorithm

Counties are matched using a **weighted Euclidean distance** algorithm. For each county, we compute a feature vector across demographic, economic, and industry variables. The distance between two counties is calculated as:

```
distance = || (county_A - county_B) * weights ||
```

Lower distance = more similar counties.

The algorithm then ranks all counties in the candidate pool by distance and returns the closest matches after applying filters (see below).

### Feature Weighting

Not all features are treated equally. The weighting system reflects deliberate methodological choices about what constitutes meaningful similarity:

**Racial & Ethnic Demographics (highest weight)**
Minority demographic groups — Black, Hispanic, Asian, American Indian/Alaska Native, and Native Hawaiian/Pacific Islander populations — are weighted at 50. White population is weighted at 20.

This is intentional. White population percentage is near-universal across US counties and carries less signal for identifying meaningfully distinct communities. Minority population shares are more predictive of shared lived experience, economic conditions, and policy context, and therefore receive higher weight in the matching process.

Dynamic adjustments are also applied based on the demographic profile of the selected county:
- A racial group making up 40%+ of the local population receives a 5x weight boost
- A racial group making up 20%+ receives a 3x boost
- A racial group making up 7% or less receives a negative weight adjustment, deprioritizing counties where that group is nearly absent and avoiding spurious matches based on marginal demographic overlap

**Industry Composition (moderate weight)**
Primary and secondary industries are weighted at 10. Shared economic structure is an important dimension of community similarity — a manufacturing county and a tourism county may look demographically similar but have fundamentally different economic and social dynamics.

**Geography & Population (lower weight)**
Rural/urban classification is weighted at 10. Raw population is weighted at 5, with additional threshold-based logic to ensure large metropolitan counties are compared against appropriately sized peers rather than being matched to small rural counties.

### Population Filtering

For counties with populations under 1 million, the algorithm restricts matches to counties within 5 percentile points of the selected county's population percentile. This prevents small rural counties from being matched to large urban ones solely on demographic grounds.

For very large counties (over 1 million), explicit population range thresholds are applied to keep comparisons meaningful.

### Income Filter

Mirror counties are required to have a median household income at least 10% greater than the selected county. This is a deliberate design choice: the tool is oriented toward identifying counties that are performing *better* on economic outcomes, so that policymakers can study and learn from their approaches rather than simply confirming shared disadvantage.

---

## Culture Analysis

The Culture Analysis page uses a large language model (GPT-4o Mini) to generate qualitative analysis comparing the selected county and its mirrors. The model is given the demographic and economic data for the selected county and its top mirror counties and prompted to identify meaningful similarities and differences across cultural, historical, economic, and social dimensions.

The LLM is instructed to:
- Use only the provided data for any quantitative claims
- Draw on historical, cultural, and economic context for qualitative analysis
- Identify the best mirror county for policy collaboration and explain why
- Avoid fabricating statistics or rankings not present in the data

This feature is designed to complement the quantitative matching — adding context that raw numbers cannot capture.

---

## Manual Controls

Advanced users can disable the default algorithmic assumptions via the "Turn off Algorithmic Assumptions" toggle in the sidebar. This enables:

- Custom population min/max filters
- Manual race weight multiplier
- Manual industry weight multiplier
- Adjustable number of mirror counties returned (k)

This is useful for researchers who want to explore sensitivity to weighting choices or test specific hypotheses about community similarity.

---

## Limitations & Caveats

- **Data recency** — demographic and economic data reflects the most recent available Census / ACS estimates and may not capture very recent population shifts
- **County-level aggregation** — county averages can obscure significant within-county variation, particularly in large or geographically diverse counties
- **Weighting is opinionated** — the default weights reflect deliberate methodological choices that prioritize minority demographic representation and economic opportunity gaps; researchers with different priorities may want to use the manual controls
- **Income filter directionality** — the 10% income floor means the tool surfaces upward comparisons only; it is not designed to find counties performing similarly or worse
- **LLM analysis** — qualitative analysis is AI-generated and should be treated as a starting point for research, not a definitive characterization of any community

---

## About American Inequality

The American Inequality Project publishes data-driven research on economic inequality, healthcare access, demographic shifts, and social disparities in the United States. Founded by Jeremy Ney, the project's work has been featured in The New York Times, NPR, TIME, and BBC, and informs college curricula at Harvard, MIT, Columbia, and Georgetown.

📰 [Read on Substack](https://americaninequality.substack.com/?utm_campaign=profile_chips)

---

## About the Project

**American Inequality** was founded by **Jeremy Ney**, economist and author. 
Jeremy is the primary author and researcher behind the project's publications on 
inequality, healthcare, and opportunity in America.

🔗 [Follow Jeremy on Substack](https://substack.com/@jeremybney) | [LinkedIn](https://www.linkedin.com/in/jeremy-ney/)


**Michael Tiede** is Head Data Scientist at American Inequality, where he designed 
and built the Mirror Counties app and matching algorithm. He also writes 
*The Dividing Line* series on geography, race, and economic opportunity.

🔗 [Read The Dividing Line](https://substack.com/@michaeltiede/posts) | [LinkedIn](https://www.linkedin.com/in/michaeltiede/)
