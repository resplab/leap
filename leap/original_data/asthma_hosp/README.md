# Asthma Hospitalization Data

The data in this dataset were obtained from the Hospital Morbidity Database, which is
maintained by the Canadian Institute for Health Information (CIHI).

More details on this dataset can be found in this paper:

https://www.sciencedirect.com/science/article/pii/S1081120622005403#bib0009

> The database provides full geographic coverage of inpatient admissions across Canada.
> Each record contains dates of admission and discharge, biological sex, age, and up to 25
> discharge diagnoses in the International Classification of Diseases (ICD), ninth or tenth
> revisions, depending on when the tenth revision was adopted in each province. One of the
> diagnostic codes is designated as the primary (most responsible) diagnosis. We had access to all
> hospitalization records containing at least 1 diagnostic code for asthma
> (ICD-9 493 excluding 493.2 or ICD-10 J45/J46 [ICD-10-CA J45]) in the period from 2002 to 2017.
> We identified asthma-related hospital admissions as those with asthma as the primary diagnosis.

In our model, we define a **severe exacerbation** as one which requires hospital admission
(or in our notation, `severity_level = 4`). Hence, this dataset contains information on
observed severe exacerbations in each province.

<div class="note" style='padding:15px; background-color:#E9D8FD; color:#69337A'>
<span>
<p style='text-align:left'>
<b>Note:</b></p>
<p>
Nunavut, Northwest Territories, and Yukon Territory have been combined into a single category,
"TR".
</p>
</span>
</div>


## File Metadata

The columns in the following 5 files are the same, just with different values:

| column    | description |
| -------- | ------- |
| `fiscal_year`  | the year the data was collected |
| `N` | the value (e.g. count, rate) for all ages and sexes in that year |
| `M` | the value for all ages who are male in that year |
| `F` | the value for all ages who are female in that year |
| `0` | the value for all sexes aged 0 in that year |
| `1` | the value for all sexes aged 1 in that year |
| ... | ... |
| `90` | the value for all sexes aged 90 in that year |
| `90+` | the value for all sexes aged over 90 in that year |
| `F_0` | the value for females aged 0 in that year |
| ... | ... |
| `F_90+` | the value for females aged over 90 in that year |
| `M_0` | the value for males aged 0 in that year |
| ... | ... |
| `M_90+` | the value for males aged over 90 in that year |

### `tab1_count.csv`

The value in this table is the number of people who were hospitalized with asthma. For example, in
the category `F_90+`, the value would be the number of people who were hospitalized with asthma
who are female and over 90 during the given year.

### `tab1_lower.csv`

The value in this table is the lower error bar for the hospitalization rate per 100 000 people.
For example, in the category `F_90+`, the value would be lower error bar for people who are female
and over 90 during the given year.


### `tab1_upper.csv`

The value in this table is the upper error bar for the hospitalization rate per 100 000 people.
For example, in the category `F_90+`, the value would be upper error bar for people who are female
and over 90 during the given year.


### `tab1_N.csv`

The value in this table is the total number of people in the category. For example, in the
category `F_90+`, the value would be the total number of people for whom data was collected
who are female and over 90 during the given year.


### `tab1_rate.csv`

The value in this table is the hospitalization rate per 100 000 people. For example, in the
category `F_90+`, the value would be the number of people hospitalized who are female and
over 90 during the given year. This can be calculated:

```math
\begin{align}
\text{rate} &= \dfrac{\text{count}}{N} \times 100000
\end{align}
```