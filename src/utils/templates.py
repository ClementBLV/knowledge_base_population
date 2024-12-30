WN_LABELS = [
    "_hypernym",
    "_derivationally_related_form",
    "_instance_hypernym",
    "_also_see",
    "_member_meronym",
    "_synset_domain_topic_of",
    "_has_part",
    "_member_of_domain_usage",
    "_member_of_domain_region",
    "_verb_group",
    "_similar_to",
]


WN_LABEL_TEMPLATES = {  # in the future we will add the indirect relations
    "_hypernym": [
        "{obj} specifies {subj}",
        "{subj} generalize {obj}",
    ],
    "_derivationally_related_form": [
        "{obj} derived from {subj}",
        "{subj} derived from {obj}",
    ],
    "_instance_hypernym": [
        "{obj} is a {subj}",
        "{subj} such as {obj}",
    ],
    "_also_see": [
        "{obj} is seen in {subj}",
        "{subj} has {obj}",
    ],
    "_member_meronym": [
        "{obj} is the family of {subj}",
        "{subj} is a member of {obj}",
    ],
    "_synset_domain_topic_of": [
        "{obj} is a topic of {subj}",
        "{subj} is the context of {obj}",
    ],
    "_has_part": [
        "{obj} contains {subj}",
        "{subj} is a part of {obj}",
    ],
    "_member_of_domain_usage": [
        "{obj} is a domain usage of {subj}",
        "X",
    ],
    "_member_of_domain_region": [
        "{obj} is the domain region of {subj}",
        "{subj} belong to the regieon of {obj}",
    ],
    "_verb_group": [
        "{obj} is synonym to {subj}",
        "{subj} is synonym to {obj}",
    ],
    "_similar_to": [
        "{obj} is similar to {subj}",
        "{subj} similar to {obj}",
    ],
}


FORBIDDEN_MIX = {
    "_hypernym": [
        "{obj} specifies {subj}",
        "{subj} generalize {obj}",
        "{obj} derived from {subj}",
        "{subj} derived from {obj}",
        "{obj} is a {subj}",
        "{subj} such as {obj}",
    ],
    "_derivationally_related_form": [
        "{obj} derived from {subj}",
        "{subj} derived from {obj}",
        "{obj} specifies {subj}",
        "{subj} generalize {obj}",
        "{obj} is a {subj}",
        "{subj} such as {obj}",
    ],
    "_instance_hypernym": [
        "{obj} is a {subj}",
        "{subj} such as {obj}",
        "{obj} specifies {subj}",
        "{subj} generalize {obj}",
        "{obj} derived from {subj}",
        "{subj} derived from {obj}",
    ],
}

templates_direct = [
    "{obj} specifies {subj}",
    "{obj} derived from {subj}",
    "{obj} is a {subj}",
    "{obj} is seen in {subj}",
    "{obj} is the family of {subj}",
    "{obj} is a topic of {subj}",
    "{obj} contains {subj}",
    "{obj} is a domain usage of {subj}",
    "{obj} is the domain region of {subj}",
    "{obj} is synonym to {subj}",
    "{obj} is similar to {subj}",
]

template_indirect = [
    "{subj} generalize {obj}",
    "{subj} derived from {obj}",
    "{subj} such as {obj}",
    "{subj} has {obj}",
    "{subj} is a member of {obj}",
    "{subj} is the context of {obj}",
    "{subj} is a part of {obj}",
    "{subj} is a domain usage of {obj}",
    "{subj} belong to the regieon of {obj}",
    "{subj} is synonym to {obj}",
    "{subj} similar to {obj}",
]

FB_LABEL_TEMPLATES = {
    "position_s football_historical_roster_position american_football players sports_position sports ": [
        "{subj} and {obj} are two linked roster position ",
        "{subj} and {obj} are two linked roster position "
    ],
    "currency dated_money_value measurement_unit assets business_operation business ": [
        "{subj} measures its assets and conducts its business operations using the {obj} currency",
        "{obj} is the currency used by {subj} to measure its assets and conduct its business operations"
    ],
    "edited_by film ": [
        "the film {subj} was edited by {obj}",
        "{obj} was the editor of {subj}"
    ],
    "location place_lived places_lived person people ": [
        "{subj} lived in {obj}",
        "{obj} was the location where {subj} lived"
    ],
    "films film_subject film ": [
        "{subj} is the subject of the film {obj}",
        "{obj} depicted the subject of {subj}"
    ],
    "film_distribution_medium film_film_distributor_relationship distributors film ": [
        "{subj} was distributed on the following medium {obj} ",
        "{obj} was the medium on which {subj} was distributed"
    ],
    "nominated_for award_nomination nominees award_category award ": [
        "{subj} was the award category where the movie {obj} was nominee",
        "{obj} was nomenee in the award category {subj}"
    ],
    "military_combatant_group combatants military_conflict military ": [
        "The war of {subj} involved {obj}",
        "Combatant group from {obj} were involved during the {subj} "
    ],
    "cast_members snl_season_tenure saturdaynightlive base seasons snl_cast_member saturdaynightlive base ": [
        "{subj} and {obj} were cast of Saturday Night Live for overlapping seasons",
        "{subj} and {obj} were cast of Saturday Night Live for overlapping seasons"
    ],
    "medal olympic_medal_honor medals_awarded olympic_games olympics ": [
        "During {subj} the honor medal was {obj}",
        "{obj} was the honor medal of {subj} "
    ],
    "administrative_parent administrative_area schema aareas base ": [
        "The {subj} belongs geographically to {obj}",
        "{obj} is the administrative parent of {subj}"
    ],
    "position baseball_roster_position baseball roster sports_team sports ": [
        "team {subj} has the baseball roster position {obj} ",
        "{obj} is a baseball roster position present in {subj} "
    ],
    "produced_by film ": [
        "the film {subj} was produced by {obj}",
        "{obj} had produced the film {subj} "
    ],
    "film_format film ": [
        "The film {subj} is made in the format {obj}",
        "{obj} is the film format of {subj}"
    ],
    "producer_type tv_producer_term programs_produced tv_producer tv ": [
        "{subj} was the following type of tv producer {obj}",
        "{obj} was the tv producer type of {subj}"
    ],
    "film_release_distribution_medium film_regional_release_date release_date_s film ": [
        "{subj} was released on {obj}",
        "{obj} is the support for {subj}"
    ],
    "film_sets_designed film_set_designer film ": [
        "{subj} designed the film set of {obj} ",
        "{obj} film set were designed by {subj} "
    ],
    "participant popstra base canoodled celebrity popstra base ": [
        "{subj} and {obj} ones canoodled each others",
        "{obj} and {subj} ones canoodled each others"
    ],
    "country_of_origin tv_program tv ": [
        "tv program {subj} originates from {obj}",
        "{obj} is the country from which the tv program {subj} originates"
    ],
    "artist record_label music ": [
        "{subj} is the music record label of {obj}",
        "{obj} has {subj} as its record label"
    ],
    "sport sports_team sports ": [
        "{subj} is a {obj} sports team",
        "{obj} has professional sports team like {subj} "
    ],
    "contact_category phone_sandbox schemastaging base phone_number organization_extra schemastaging base ": [
        "{subj} have its organization phone sandbox in {obj}",
        "{obj} is the contact category phone sandbox of the organization {subj}"
    ],
    "sibling sibling_relationship sibling_s person people ": [
        "{subj} is a sibling of {obj} ",
        "{obj} is a sibling of {subj} "
    ],
    "organization organization_membership member_of organization_member organization ": [
        "{subj} is a memeber of the organisation of the {obj}",
        "{obj} is an organization with member like {subj}"
    ],
    "administrative_area_type administrative_area schema aareas base ": [
        "{subj} has the administrative area type {obj}",
        "{obj} is the administrative area type of {subj}"
    ],
    "actor regular_tv_appearance regular_cast tv_program tv ": [
        "One of the cast of {subj} was {obj}",
        "The actor {obj} made regular tv appearance in the program {subj}"
    ],
    "legislative_sessions government_position_held members legislative_session government ": [
        "Members of the legislative branch convened during the respective legislative sessions of both {subj} and {obj}",
        "Members of the legislative branch convened during the respective legislative sessions of both {subj} and {obj}"
    ],
    "special_performance_type performance actor film ": [
        "actor {subj} has the special performance {obj}",
        "special performance {obj} was performed by the actor {subj}"
    ],
    "school sports_league_draft_pick draft_picks professional_sports_team sports ": [
        "{subj} team selected players from {obj} during the draft picks process ",
        "during the draft picks process {obj} has players picked up for {subj} team"
    ],
    "category webpage topic common ": [
        "{subj} has the category webpage topic common {obj}",
        "{obj} is the category webpage topic common {subj}"
    ],
    "school_type educational_institution education ": [
        "{subj} is the following type of university {obj}",
        "{obj} is the type of university like {subj}"
    ],
    "film_festivals film ": [
        "{subj} was screened at {obj}",
        "during {obj} festival, {subj} was screened"
    ],
    "participant popstra base friendship celebrity popstra base ": [
        "{subj} and {obj} share a friendship thanks to their pop culture implication ",
        "{obj} and {subj} share a friendship thanks to their pop culture implication "
    ],
    "symptom_of symptom medicine ": [
        "{subj} is a symptom of {obj}",
        "{obj} causes symptoms like {subj}"
    ],
    "cause_of_death people ": [
        "{subj} was the cause of death of {obj}",
        "{obj} died of a {subj}"
    ],
    "currency dated_money_value measurement_unit local_tuition university education ": [
        "The local tuition of the the university {subj} is in the currency {obj}",
        "{obj} is the currency used to pay the local tuition for the university of {subj}"
    ],
    "film_art_direction_by film ": [
        "the art director of the film {subj} was {obj}",
        "{obj} was the art director of the film {subj}"
    ],
    "split_to gardening_hint dataworld ": [
        "{subj} split to gardening hint dataworld {obj} ",
        "{obj} split to gardening hint dataworld {subj} "
    ],
    "citytown mailing_address location headquarters organization ": [
        "{subj} headquarters are located in {obj}",
        "{obj} is the mailing address location of the headquarters of the organization {subj} "
    ],
    "director film ": [
        "{subj} was the director of the film {obj}",
        "{obj} was directed by {subj}"
    ],
    "person personal_film_appearance personal_appearances film ": [
        "the film {subj} hosted the {obj} ",
        "{obj} made an appearance in the film {subj} "
    ],
    "student students_graduates educational_institution education ": [
        "{subj} was the university of {obj}",
        "{obj} was ones a student of {subj}"
    ],
    "legislative_sessions government_position_held government_positions_held politician government ": [
        "{subj} held a government position participating in legislative sessions in {obj}",
        "During {obj} , the politician {subj} participated to legislative sessions"
    ],
    "artists genre music ": [
        "{subj} is the music genre of the artist {obj} ",
        "{obj} is an artist playing {subj} music genre"
    ],
    "company employment_tenure people_with_this_title job_title business ": [
        "{subj} is a job title in {obj}",
        "{obj} employs people with the following title job {subj}"
    ],
    "production_companies film ": [
        "{subj} was produced by {obj} company",
        "{obj} is the production company of the film {subj}"
    ],
    "award_honor awards_won award_winner award ": [
        "{subj} and {obj} are award winners",
        "{obj} and {subj} are award winner"
    ],
    "student students_majoring field_of_study education ": [
        "{subj} was the major field of study of {obj}",
        "{obj} majored in the field {subj}"
    ],
    "participating_countries olympic_games olympics ": [
        "during {subj} the country {obj} participated",
        "{obj} was one of the countries which participated to {subj}"
    ],
    "language film ": [
        "{subj} is a film in {obj} language",
        "{obj} is the language of the film {subj}"
    ],
    "country mailing_address location headquarters organization ": [
        "{subj} has its country mailing address located in {obj}",
        "{obj} is the mailing address location of {subj} country"
    ],
    "tv_program tv_program_writer_relationship tv_programs tv_writer tv ": [
        "{subj} is the tv program writer of {obj} ",
        "{obj} was written by {subj}"
    ],
    "program tv_producer_term programs_produced tv_producer tv ": [
        "{subj} was the tv producer of {obj}",
        "{obj} was produced by {subj}"
    ],
    "sports olympic_games olympics ": [
        "{subj} had {obj} as one of its olympic sports ",
        "{obj} was an olympic sport for the {subj}"
    ],
    "place_of_burial deceased_person people ": [
        "{subj} is buried in {obj}",
        "{obj} is the burial place of {subj}"
    ],
    "colors educational_institution education ": [
        "the color of {subj} of the educational institution {obj}",
        "{obj} is the color of the educational institution {subj}"
    ],
    "official_language country location ": [
        "{subj} official language is {obj}",
        "{obj} is the official language in {subj}"
    ],
    "taxonomy taxonomy_entry random tsegaran user entry taxonomy_subject random tsegaran user ": [
        "{subj} falls withing the taxonomy of the {obj}",
        "{obj} has {subj} within its taxonomie"
    ],
    "currency dated_money_value measurement_unit gdp_nominal_per_capita statistical_region location ": [
        "{subj} uses the {obj} as a currency reference point for measuring the nominal GDP",
        "{obj} is the currency used as a reference to measure the nominal GDP of {subj}"
    ],
    "position sports_team_roster sports current_roster football_team american_football ": [
        "{subj} has a {obj} in their current roster",
        "{obj} is one of the roster of {subj} "
    ],
    "jurisdiction_of_office government_position_held government_positions_held politician government ": [
        "the politician {subj} was under the jurisdiction of {obj}",
        "{obj} had the politician {subj}"
    ],
    "service_location phone_sandbox schemastaging base phone_number organization_extra schemastaging base ": [
        "{subj} has it service base located in {obj}",
        "{obj} is the location where {subj} has its service based"
    ],
    "person tv_regular_personal_appearance tv_regular_personal_appearances non_character_role tv ": [
        "{obj} made regular tv appearance as a {subj} character ",
        "{obj} made regular tv appearance as a {subj} character "
    ],
    "jurisdiction_of_office government_position_held officeholders government_office_category government ": [
        "{subj} is the government office category of the position held by officeholders in {obj} ",
        "in the jurisdiction of {obj} the government office position is {subj}"
    ],
    "service_language phone_sandbox schemastaging base phone_number organization_extra schemastaging base ": [
        "{subj} has its base phone number in {obj}",
        "{obj} is the base phone number of {subj}"
    ],
    "place_of_death deceased_person people ": [
        "{subj} died in {obj}",
        "{obj} was the place of death of {subj}"
    ],
    "position football_roster_position current_roster football_team soccer ": [
        "in the current roster of the {subj} football team, {obj} is one of the position",
        "{obj} is one of the position in the roster of {subj} football team"
    ],
    "friend friendship celebrity_friends celebrity celebrities ": [
        "{subj} and {obj} are two celebrity friends",
        "{obj} and {subj} are two celibrity friends"
    ],
    "exported_to imports_and_exports places_exported_to statistical_region location ": [
        "{subj} imports goods from {obj} ",
        "{obj} imports goods from {subj}"
    ],
    "capital country location ": [
        "{subj} is the country to which {obj} is the capital ",
        "{obj} is the capital of {subj}"
    ],
    "major_field_of_study students_majoring field_of_study education ": [
        "{subj} is the major field of study for {obj}",
        "{obj} is a sub field of study of {subj}"
    ],
    "diet practicer_of_diet eating base ": [
        "{subj} has a diet based on {obj}",
        "{obj} is the diet eating base of {subj} "
    ],
    "contains location ": [
        "{subj} contains the location {obj} ",
        "{obj} is a place of {subj}"
    ],
    "registering_agency non_profit_registration registered_with non_profit_organization organization ": [
        "{subj} is register with {obj} agency as a non profit",
        "{obj} agency has register {subj} as a non profit"
    ],
    "instrumentalists instrument music ": [
        "{subj} was the instrument of {obj}",
        "{obj} was a instrumentalists of {subj}"
    ],
    "medal olympic_medal_honor medals_won olympic_participating_country olympics ": [
        "{subj} won the {obj} during the olympic games",
        "{obj} was won by {subj} during the olympics games"
    ],
    "state bibs_location biblioness base ": [
        "{subj} is the state location of the biblioness base for {obj}",
        "{obj} biblioness base has its state location in {subj}"
    ],
    "film_production_design_by film ": [
        "In the film {subj} the production design was done by {obj} ",
        "{obj} was the production designer in {subj} "
    ],
    "time_zones location ": [
        "{subj} follow the time zone location {obj} ",
        "{obj} is the time zone of {subj}"
    ],
    "disciplines_or_subjects award_category award ": [
        "{subj} is a award for {obj}",
        "{obj} are subject to the following award {subj}"
    ],
    "month travel_destination_monthly_climate climate travel_destination travel ": [
        "{subj} is a good destination to travel in {obj} ",
        "{obj} is the best month to travel in {subj} "
    ],
    "notable_people_with_this_condition disease medicine ": [
        "{subj} is a condition disease which affected famous person {obj}",
        "{obj} was a famous person with {subj}"
    ],
    "adjoins adjoining_relationship adjoin_s location ": [
        "{subj} adjoins {obj} ",
        "{obj} adjoins {subj}"
    ],
    "program_creator tv_program tv ": [
        "tv program {subj} was created by {obj}",
        "{obj} was the creator of {subj}"
    ],
    "cinematography film ": [
        "In the film {subj}, the cinematography was handled by {obj}",
        "{obj} handled the cinematography for the film {subj}  "
    ],
    "currency dated_money_value measurement_unit net_worth person_extra schemastaging base ": [
        "{subj}'s net worth to gauge financial success is measured in {obj}",
        "{obj} is the currency used to gauge financial success of {subj}"
    ],
    "category_of award_category award ": [
        "{subj} is a category of the award {obj}",
        "{obj} has the category {subj}"
    ],
    "award_nomination award_nominations award_nominee award ": [
        "Both {subj} and {obj} have been recognized as nominees",
        "Both {obj} and {subj} have been recognized as nominees"
    ],
    "educational_institution educational_institution_campus education ": [
        "{subj} is an educational institution with its campus located in {obj}",
        "{obj} is the campus location of the educational institution {subj}"
    ],
    "costume_design_by film ": [
        "costumes in {subj} were designed by {obj}",
        "{obj} designed the costumes in {subj}"
    ],
    "team sports_league_participation teams sports_league sports ": [
        "{subj} is a sport team group to which {obj} team belongs",
        "the {obj} participates to the sport team league {subj}"
    ],
    "student people_with_this_degree educational_degree education ": [
        "{subj} is the type of degree {obj}",
        "{obj} has a degree in {subj}"
    ],
    "languages person people ": [
        "{subj} speaks {obj}",
        "{obj} was the language {subj} spokes"
    ],
    "program tv_network_duration programs tv_network tv ": [
        "{subj} broadcasted the tv program {obj}",
        "{obj} was broadcasted by {subj} "
    ],
    "colors sports_team sports ": [
        "{subj} color sport team is {obj}",
        "{obj} is the color of the sport team {subj}"
    ],
    "languages tv_program tv ": [
        "TV program {subj} was aired in {obj}",
        "{obj} was the language of the TV program {subj}"
    ],
    "family instrument music ": [
        "{subj} belongs to the instrument family of {obj}",
        "{obj} is the family instrument of {subj}"
    ],
    "currency dated_money_value measurement_unit international_tuition university education ": [
        "{subj} has its international tuition in currency {obj}",
        "{obj} is the currency for the international tuition fees in {subj}"
    ],
    "place_founded organization ": [
        "{subj} was founded in the place {obj}",
        "{obj} is the place where {subj} was founded "
    ],
    "olympics olympic_athlete_affiliation athletes olympic_participating_country olympics ": [
        "{subj} participated at the {obj}",
        "during {obj} the country {subj} participated"
    ],
    "capital administrative_area schema aareas base ": [
        "{subj} has its administrative capital in {obj}",
        "{obj} is the capital of the administrative area of {subj}"
    ],
    "crewmember film_crew_gig other_crew film ": [
        "{subj} had {obj} within its crew memebers",
        "{obj} was one of the crew memeber for the film {subj} "
    ],
    "recording_contribution guest_performances performance_role music ": [
        "{subj} recording contribution guest performances performance role music {obj}",
        "{obj} recording contribution guest performances performance role music {subj}"
    ],
    "producer_type tv_producer_term tv_producer tv_program tv ": [
        "{subj} is tv producer of the following type {obj} ",
        "{obj} is the type of tv producer of {subj}"
    ],
    "award award_honor awards_won award_winning_work award ": [
        "{subj} won the award {obj}",
        "{obj} is an award won by the {subj} "
    ],
    "performance_role recording_contribution contribution artist music ": [
        "{subj} has the performance role recording contribution of {obj}",
        "{obj} was the recording contribution role of {subj} "
    ],
    "music film ": [
        "the music film {subj} was composed by {obj}",
        "{obj} participated in the music film {subj}"
    ],
    "celebrities_impersonated celebrity_impressionist americancomedy base ": [
        "{subj} has impersonated {obj} ",
        "{obj} was impersonated by {subj}"
    ],
    "language dubbing_performance dubbing_performances actor film ": [
        "{subj} is a dubbing actor in {obj}",
        "{obj} is the dubbing language performed by the actor {subj}"
    ],
    "risk_factors disease medicine ": [
        "{subj} risks is increased by the following factor {obj}",
        "{obj} is a risk factor of {subj}"
    ],
    "currency dated_money_value measurement_unit estimated_budget film ": [
        "the budget of {subj} was estimated in {obj}",
        "{obj} was the currency used to estimate the budget of {subj}"
    ],
    "film film_film_distributor_relationship films_distributed film_distributor film ": [
        "{subj} is the film ditributor of {obj}",
        "The film {obj} was distributed by {subj}"
    ],
    "country film ": [
        "{subj} is the country of the film {obj}",
        "{obj} is the country of the film {subj}"
    ],
    "award award_nomination award_nominations award_nominee award ": [
        "{subj} had an award nomination at the award {obj}",
        "{obj} was the award were {subj} was a nominee"
    ],
    "team ncaa_tournament_seed marchmadness base seeds ncaa_basketball_tournament marchmadness base ": [
        "{subj} was a basketball tornament where {obj} participated as a seeded team",
        "{obj} basketball team participated in the {subj} as a seeded team "
    ],
    "romantic_relationship sexual_relationships celebrity celebrities ": [
        "{subj} was involved in a romantic and sexual relationship with the celebrity {obj}",
        "{obj} was involved in a romantic and sexual relationship with the celebrity {subj}"
    ],
    "gender person people ": [
        "{subj} has the gender of a {obj}",
        "{obj} is the gender of {subj}"
    ],
    "position sports_team_roster sports current_roster football_team soccer ": [
        "The team {subj} has the following position {obj} ",
        "{obj} is a football position of the {subj}"
    ],
    "ceremony award_honor winners award_category award ": [
        "{subj} is an award cathegory of {obj}",
        "{obj} is the ceremony award for the {subj}"
    ],
    "state_province_region mailing_address location headquarters organization ": [
        "{subj} has its headquarter mailing address in the state province of {obj}",
        "{obj} is the state province mailing address of {subj}"
    ],
    "adjustment_currency adjusted_money_value measurement_unit gdp_real statistical_region location ": [
        "{subj} uses {obj} as a reference for currency adjustments",
        "{obj} is the currency used in {subj} for its own currency adjustment"
    ],
    "type_of_appearance personal_film_appearance films person_or_entity_appearing_in_film film ": [
        "{subj} had made appearance in {obj}",
        "{obj} is the type of public appearance {subj} made "
    ],
    "ethnicity people ": [
        "{subj} is the ethnicity of {obj}",
        "{obj} belong to the ethnicity of {subj}"
    ],
    "program tv_regular_personal_appearance tv_regular_appearances tv_personality tv ": [
        "{subj} made regular tv appearance in the show {obj} ",
        "{obj} frequently hosted the celebrity {subj} "
    ],
    "film_release_region film_regional_release_date release_date_s film ": [
        "the region where the film {subj} was released is {obj}",
        "{obj} is a region where the film {subj} was released"
    ],
    "position basketball_roster_position basketball roster sports_team sports ": [
        "the basketball team {subj} has the roster {obj}",
        "{obj} is a roster in the basketball team {subj}"
    ],
    "season baseball_team_stats team_stats baseball_team baseball ": [
        "{subj} baseball team made major team stats during the baseball season {obj} ",
        "during the baseball season {obj} the baseball team {subj} made major team stats "
    ],
    "titles netflix_genre media_common ": [
        "{subj} is the media genre of {obj}",
        "the netflix show {obj} is a {subj}"
    ],
    "genre tv_program tv ": [
        "The tv program {subj} has the following genre {obj}",
        "{obj} is the genre of the tv program {subj} "
    ],
    "profession person people ": [
        "{subj} has the profession of {obj}",
        "{obj} is the profession of {subj}"
    ],
    "place hud_county_place location ": [
        "{subj} is a place hud county location of {obj}",
        "{subj} is a place hud county location of {obj}"
    ],
    "basic_title government_position_held government_positions_held politician government ": [
        "{subj} has the governement position of {obj}",
        "As a member of {obj} , {subj} held various governement positions"
    ],
    "nutrient nutrition_fact nutrients food ": [
        "{subj} contains the nutrient {obj}",
        "{obj} is a nutrient in {subj} "
    ],
    "currency dated_money_value measurement_unit gdp_nominal statistical_region location ": [
        "{subj} gdp is measured in {obj} ",
        "{obj} is the currency used to measure the gdp of {subj}"
    ],
    "legislative_sessions government_position_held members governmental_body government ": [
        "{subj} was the governemental body during {obj}",
        "{subj} was the governemental body during {obj}"
    ],
    "prequel film ": [
        "the film {subj} has the prequel {obj}",
        "{obj} is the prequel of {subj}"
    ],
    "currency dated_money_value measurement_unit endowment endowed_organization organization ": [
        "{subj} is paid in {obj}",
        "{obj} is the currency used to pay {subj}"
    ],
    "county hud_county_place location ": [
        "{subj} is located in the county of {obj} ",
        "{obj} is the county were {subj} is located"
    ],
    "origin artist music ": [
        "The music artist {subj} originated from {obj} ",
        "{obj} is the origin location of the artist {subj}"
    ],
    "religion religion_percentage religions statistical_region location ": [
        "{subj} has an important religious percentage of {obj}",
        "{obj} is significative religions distribution in the region of {subj} "
    ],
    "team sports_team_roster teams pro_athlete sports ": [
        "{subj} was a pro athlete in {obj}",
        "{obj} was the team of {subj}"
    ],
    "film performance film_performance_type special_film_performance_type film ": [
        "{subj} is the type of actor performance for the film {obj}",
        "the film {obj} has the type of actor performance {subj}"
    ],
    "draft sports_league_draft_pick draft_picks professional_sports_team sports ": [
        "New player for baseball league {subj} were selected during {obj}",
        "During {obj} the sport league {subj} selected new players"
    ],
    "countries_spoken_in human_language language ": [
        "{subj} is spoken in {obj}",
        "{obj} spokes {subj}"
    ],
    "source dated_integer measurement_unit estimated_number_of_mortgages hud_foreclosure_area location ": [
        "In {subj} area the source used to estimate the number of mortgages hub foreclosure is {obj} ",
        "{obj} is the source for the measurements and estimations of the number of mortgages hub foreclosure in {subj}"
    ],
    "currency dated_money_value measurement_unit revenue business_operation business ": [
        "{subj} makes its revenue in {obj} ",
        "{obj} is the currency used to evaluate the revenues of {subj}"
    ],
    "peer_relationship peers influence_node influence ": [
        "{subj} and {obj} shared a peer relationship",
        "{obj} and {subj} shared a peer relationship"
    ],
    "instance_of_recurring_event event time ": [
        "{subj} is a recurring event of {obj}",
        "{obj} is the main event of {subj}"
    ],
    "currency dated_money_value measurement_unit rent50_2 statistical_region location ": [
        "{subj} uses the currency {obj} ",
        "{obj} is the currency used in {subj}"
    ],
    "spouse marriage spouse_s person people ": [
        "{subj} is the spouse of {obj}",
        "{obj} has his / her spouse named {subj}"
    ],
    "religion person people ": [
        "{subj} practice {obj} religion",
        "{obj} is the religion of {subj}"
    ],
    "featured_film_locations film ": [
        "The film {subj} featured {obj} location ",
        "{obj} was the location featured in {subj} "
    ],
    "seasonal_months produce_availability localfood base produce_available seasonal_month localfood base ": [
        "{subj} and {obj} are both seasonal months to produce local food ",
        "{subj} and {obj} are both seasonal months to produce local food "
    ],
    "group group_membership membership group_member music ": [
        "{subj} is a memeber of the music group {obj}",
        "{obj} is the music group of {subj}"
    ],
    "mode_of_transportation transportation how_to_get_here travel_destination travel ": [
        "to get in {subj} the transportation mode is {obj}",
        "{obj} is the transportation mode to get in {subj}"
    ],
    "story_by film ": [
        "The film {subj} was written by {obj}",
        "{obj} wrote the film {subj}"
    ],
    "entity_involved event culturalevent base ": [
        "the cultural event {subj} involved {obj}",
        "{obj} were involved in the cultural event {subj} "
    ],
    "written_by film ": [
        "the film {subj} was written by {obj}",
        "{obj} wrote the movie {subj}"
    ],
    "fraternities_and_sororities university education ": [
        "{subj} hosts the sorority {obj}",
        "{obj} is a sorority of {subj}"
    ],
    "second_level_divisions country location ": [
        "{subj} has second level division such as {obj} ",
        "{obj} belongs to the second level country division of the {subj}"
    ],
    "athlete pro_sports_played pro_athletes sport sports ": [
        "{subj} had pro athlete like {obj} ",
        "{obj} was a pro athlete in {subj}"
    ],
    "nominated_for award_nomination award_nominations award_nominee award ": [
        "{subj} received an award nomination for the movie {obj}",
        "the movie {obj} had {subj} nominated for an award"
    ],
    "country olympic_athlete_affiliation athletes olympic_sport olympics ": [
        "{subj} olympic athlete can be affiliated to the country {obj} ",
        "{obj} country has olympic athlete {subj}"
    ],
    "award_winner award_honor winners award_category award ": [
        "{subj} is an award category won by {obj} ",
        "{obj} won an award in the category {subj} "
    ],
    "country bibs_location biblioness base ": [
        "The biblioness of {subj} is located in the country {obj}",
        "{obj} is the country of the biblioness of {subj}"
    ],
    "honored_for award_honor awards_won award_winning_work award ": [
        "{subj} and {obj} were both honoured by an award",
        "{subj} and {obj} were both honoured by an award"
    ],
    "place_of_birth person people ": [
        "{subj} was born in the place {obj} ",
        "{obj} is the place of birth of {subj}"
    ],
    "interests philosopher philosophy alexander user ": [
        "philosopher {subj} had interests for {obj}",
        "{obj} was one of the interests of the philosopher {subj}"
    ],
    "teams sports_team_location sports ": [
        "The location {subj} host the sports team {obj} ",
        "The team {obj} is located in {subj}"
    ],
    "role track_contribution track_contributions artist music ": [
        "{subj} had a role trak in as a music artic with a contribution track with {obj}",
        "{obj} was an contribution of {subj} in music "
    ],
    "vacationer vacation_choice popstra base vacationers location popstra base ": [
        "{subj} is the vacation location base of the star {obj}",
        "The star {obj} spends his or her vacation in the location {subj}"
    ],
    "politician political_party_tenure politicians_in_this_party political_party government ": [
        "{subj} was the political party of {obj}",
        "{obj} is a politician from the party {subj}"
    ],
    "administrative_division administrative_division_capital_relationship capital_of capital_of_administrative_division location ": [
        "{subj} serves as the administrative division within {obj}",
        "{obj} as its administrative division in {subj} "
    ],
    "nationality person people ": [
        "the nationality of {subj} is {obj} ",
        "{obj} is the nationality of {subj}"
    ],
    "artist content broadcast ": [
        "{subj} broadcast the artist {obj}",
        "{obj} is broadcasted on {subj}"
    ],
    "member_states international_organization default_domain ktrueman user ": [
        "{subj} international organization has country memebers like {obj}",
        "{obj} is a country member of the {subj}"
    ],
    "form_of_government country location ": [
        "{subj} has the government form of a {obj} ",
        "{obj} is the government form of {subj}"
    ],
    "performance actor film ": [
        "{subj} made an apparition in the film {obj}",
        "the film {obj} ones hosted {subj} as an actor"
    ],
    "award_winner award_honor awards_won award_winning_work award ": [
        "the work {subj} of {obj} helped him or her to win an award",
        "{obj} won an award for his or her work in {subj} "
    ],
    "first_level_division_of administrative_division location ": [
        "{subj} is the first level administrative division within {obj} ",
        "{obj} has first level administrative division like {subj}"
    ],
    "actor dubbing_performance dubbing_performances film ": [
        "{subj} was dubbed by {obj}",
        "{obj} was a dubbing actor in {subj}"
    ],
    "currency dated_money_value measurement_unit gni_per_capita_in_ppp_dollars statistical_region location ": [
        "the location {subj} measure its unit gmi per capita in ppp using the {obj} currency",
        "{obj} currency is used to measure the units gmi per capita in ppp in the location {subj}"
    ],
    "executive_produced_by film ": [
        "The executive producer of the film {subj} was {obj}",
        "{obj} was the executive producer of {subj}"
    ],
    "honored_for award_honor awards_presented award_ceremony award ": [
        "during the award ceremony of {subj}, {obj} was honored",
        "{obj} was honored during the award ceremony {subj} "
    ],
    "locations event time ": [
        "{subj} took place in the location {obj} ",
        "{subj} as one of the location were {obj} took place"
    ],
    "countries_within continents locations base ": [
        "{subj} is the continent of {obj}",
        "{obj} is a country of the continent {subj}"
    ],
    "award_winner award_honor awards_presented award_ceremony award ": [
        "During the {subj} the artist {obj} won an award",
        "{obj} won a award at {subj}"
    ],
    "major_field_of_study people_with_this_degree educational_degree education ": [
        "{subj} is the major field for a degree in {obj} ",
        "A degree in {obj} has a major in {subj}"
    ],
    "dog_breed dog_city_relationship petbreeds base top_breeds city_with_dogs petbreeds base ": [
        "{subj} mostly have the pet bread {obj}",
        "The dog bread {obj} is the top breed of {subj} "
    ],
    "role group_membership regular_performances performance_role music ": [
        "{subj} can perform the music role of {obj}",
        "{obj} sound can be performed by {subj}"
    ],
    "participant popstra base dated celebrity popstra base ": [
        "{subj} and {obj} are participant in the popstart culture and dated",
        "{subj} and {obj} are participant in the popstart culture and dated"
    ],
    "school sports_league_draft_pick picks sports_league_draft sports ": [
        "during {subj}, {obj} produced draft picks selected by teams in the sports league",
        "during {subj}, {obj} produced draft picks selected by teams in the sports league"
    ],
    "role group_membership membership group_member music ": [
        "{subj} had the role of the {obj} in his music group ",
        "{obj} is the role of the member {subj} in his music group"
    ],
    "geographic_distribution ethnicity people ": [
        "{subj} are a ethinicity present in {obj}",
        "{obj} is a country with a disaspora of {subj}"
    ],
    "location_of_ceremony marriage unions_of_this_type marriage_union_type people ": [
        "{subj} ceremony takes place in {obj}",
        "In {obj} there are {subj} ceremonies "
    ],
    "sports olympic_games default_domain jg user ": [
        "During {subj} there was the sports olympic games {obj} ",
        "{obj} was one of the olympic sports during {subj}"
    ],
    "organization_relationship child organization ": [
        "{subj} is the parent organisation of {obj}",
        "{obj} is the child organisation of {subj}"
    ],
    "country administrative_division location ": [
        "{subj} has its county administrative division in  {obj}",
        "{obj} is the county administrative of the division of {subj} "
    ],
    "position sports_team_roster sports current_roster hockey_team ice_hockey ": [
        "the hockey team {subj} have {obj} i, their their roster",
        "{obj} is present in {subj} hockey team roster"
    ],
    "district_represented government_position_held members legislative_session government ": [
        "{subj} has legislative member from {obj}",
        "{obj} district had a member who had legislative position in {subj}"
    ],
    "current_club x2010fifaworldcupsouthafrica base current_world_cup_squad world_cup_squad x2010fifaworldcupsouthafrica base ": [
        "{subj} includes players from {obj} in its current World Cup squad for the 2010 FIFA World Cup in South Africa.",
        "{obj} player are included in the {subj} for the World Cup for the 2010 FIFA World Cup in South Africa."
    ],
    "team sports_team_roster sports current_team football_player soccer ": [
        "{subj} was a football soccer player in the team {obj}",
        "{obj} was the football team of the soccer player {subj} "
    ],
    "partially_contains location ": [
        "{subj} partially contains {obj}",
        "{obj} is partially contained in {subj}"
    ],
    "position football_roster_position american_football roster sports_team sports ": [
        "football team {subj} had a {obj} position in its roster",
        "{obj} is a position is the football roster of the team {subj}"
    ],
    "organizations_founded organization_founder organization ": [
        "{subj} founded the organisation {obj}",
        "{obj} was founded by {subj}"
    ],
    "position sports_team_roster players sports_position sports ": [
        "{subj} and {obj} are both roster player positions in sports teams",
        "{subj} and {obj} are both roster player positions in sports teams"
    ],
    "nominated_for award_nomination award_nominations award_nominated_work award ": [
        "{subj} and {obj} were both nominated for an award for their work",
        "{obj} and {subj} were both nominated for an award for their work"
    ],
    "combatants military_combatant_group military_conflicts military_combatant military ": [
        "Combatant from {subj} were opposed to {obj} in a military conflict",
        "Combatant from {obj} were opposed to {subj} in a military conflict"
    ],
    "county_seat us_county location ": [
        "County seat of {subj} are located in {obj}",
        "{obj} are the location for the {subj} seat"
    ],
    "industry business_operation business ": [
        "{subj} has its industry business operations in {obj}",
        "{obj} is the industry business operation of {subj}"
    ],
    "team sports_team_roster players sports_position sports ": [
        "{subj} is a postion in the team sport roster of tge {obj}",
        "sport team {obj} has the position {subj} in its roster"
    ],
    "inductee hall_of_fame_induction inductees hall_of_fame award ": [
        "{subj} is the hall of fame for {obj}",
        "{obj} is in the hall of fame {subj}"
    ],
    "parent_genre genre music ": [
        "{subj} is a musical sub genre of {obj}",
        "{obj} is the parent genre music of {subj}"
    ],
    "influenced_by influence_node influence ": [
        "{subj} was influence by the writter {obj}",
        "{obj} influence the writter {subj}"
    ],
    "film_crew_role film_crew_gig other_crew film ": [
        "The film {subj} has a crew of {obj}",
        "A crew of {obj} was present for the film {subj}"
    ],
    "olympics olympic_athlete_affiliation athletes olympic_sport olympics ": [
        "the olympic sport {subj} had athletes affiliated to {obj}",
        "during {obj} , athletes from {subj} were affiliated"
    ],
    "currency dated_money_value measurement_unit domestic_tuition university education ": [
        "{subj} has it tuiotion fees measured in {obj}",
        "{obj} is the currency unit measurement for the tuition fees of {subj}"
    ],
    "specialization_of profession people ": [
        "{subj} is a professional specialisation of {obj}",
        "the profession {obj} can specialise in {subj}"
    ],
    "olympics olympic_medal_honor medals_won olympic_participating_country olympics ": [
        "{subj} won the olympic participating honor medal during the {obj} ",
        "{obj} was when {subj} recived the participating honor medal"
    ],
    "type_of_union marriage spouse_s person people ": [
        "{subj} is united to his or her spouse through a {obj} ",
        "{obj} is the type of union between {subj} and his or her spouse"
    ],
    "major_field_of_study students_graduates educational_institution education ": [
        "{subj} gives a major in {obj}",
        "{obj} is a major given in {subj}"
    ],
    "list ranking appears_in_ranked_lists ranked_item award ": [
        "{subj} appeared in the ranking ward {obj} ",
        "{subj} ranks {obj} "
    ],
    "film_release_region film_cut runtime film ": [
        "{subj} was released in {obj} and had its runtime determined by the film cut",
        "{obj} was the released region for the film {subj} which had its runtime determined by the film cut"
    ],
    "position_s football_historical_roster_position american_football roster sports_team sports ": [
        "in the historical roster of {subj} football team, {obj} has been an important position",
        "{obj} has been an important position in the historical roster of {subj} football team"
    ],
    "role track_contribution track_performances performance_role music ": [
        "{subj} often play with {obj} in music performances",
        "{obj} often play with {subj} in music performances"
    ],
    "campuses educational_institution education ": [
        "{subj} campus location of {obj}",
        "{obj} is the campus location of {subj}"
    ],
    "film_regional_debut_venue film_regional_release_date release_date_s film ": [
        "film {subj} mad its regional debut at {obj}",
        "{obj} was the festival were the film {subj} made its regional debut "
    ],
    "group group_membership regular_performances performance_role music ": [
        "{subj} is a performance role music present in the group {obj}",
        "the music group {obj} the music role {subj} "
    ],
    "organization leadership leaders role organization ": [
        "{subj} is the leader role for the organisation of {obj}",
        "{obj} as a {subj} as leader"
    ],
    "location_of_ceremony marriage spouse_s person people ": [
        "{subj} made his or her marriage ceremony in {obj}",
        "{obj} was the location of the marriage of {subj}"
    ],
    "genre film ": [
        "The film {subj} belongs to the genre of {obj}",
        "{obj} is the genre of the film {subj} "
    ],
    "participant popstra base breakup celebrity popstra base ": [
        "celebrity {subj} and {obj} broke up",
        "celebrity {obj} and {subj} broke up"
    ],
    "company employment_tenure business employment_history person people ": [
        "{subj} was employed by {obj}",
        "{obj} employed {subj}"
    ],
    "languages_spoken ethnicity people ": [
        "ethnic minority {subj} speaks {obj}",
        "{obj} is spoken by the ethnic minority {subj}"
    ],
    "region film_film_distributor_relationship distributors film ": [
        "{subj} was distributed by a distributor from {obj}",
        "{obj} is the country from which {subj} distributor comes from"
    ],
    "institution people_with_this_degree educational_degree education ": [
        "{subj} degree can be delivered by {obj} educational institution",
        "{obj} educational institution delivers degrees like {subj} "
    ],
    "currency dated_money_value measurement_unit operating_income business_operation business ": [
        "{subj} measures its operatinf income in {obj} currency",
        "{obj} is the currency used by {subj} to measure its operating income"
    ]
}