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
        "{subj} is used in the context of {obj}",
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
    "{subj} is used in the context of {obj}",
    "{subj} belong to the regieon of {obj}",
    "{subj} is synonym to {obj}",
    "{subj} similar to {obj}",
]

FB_LABEL_TEMPLATES = {
    "position_s football_historical_roster_position american_football players sports_position sports ": [
        "{obj} and {subj} are two linked roster position ",
        "{obj} and {subj} are two linked roster position "
    ],
    "currency dated_money_value measurement_unit assets business_operation business ": [
        "{obj} measures its assets and conducts its business operations using the {subj} currency",
        "{subj} is the currency used by {obj} to measure its assets and conduct its business operations"
    ],
    "edited_by film ": [
        "the film {obj} was edited by {subj}",
        "{subj} was the editor of {obj}"
    ],
    "location place_lived places_lived person people ": [
        "{obj} lived in {subj}",
        "{subj} was the location where {obj} lived"
    ],
    "films film_object film ": [
        "{obj} is the object of the film {subj}",
        "{subj} depicted the object of {obj}"
    ],
    "film_distribution_medium film_film_distributor_relationship distributors film ": [
        "{obj} was distributed on the following medium {subj} ",
        "{subj} was the medium on which {obj} was distributed"
    ],
    "nominated_for award_nomination nominees award_category award ": [
        "{obj} was the award category where the movie {subj} was nominee",
        "{subj} was nomenee in the award category {obj}"
    ],
    "military_combatant_group combatants military_conflict military ": [
        "The war of {obj} involved {subj}",
        "Combatant group from {subj} were involved during the {obj} "
    ],
    "cast_members snl_season_tenure saturdaynightlive base seasons snl_cast_member saturdaynightlive base ": [
        "{obj} and {subj} were cast of Saturday Night Live for overlapping seasons",
        "{obj} and {subj} were cast of Saturday Night Live for overlapping seasons"
    ],
    "medal olympic_medal_honor medals_awarded olympic_games olympics ": [
        "During {obj} the honor medal was {subj}",
        "{subj} was the honor medal of {obj} "
    ],
    "administrative_parent administrative_area schema aareas base ": [
        "The {obj} belongs geographically to {subj}",
        "{subj} is the administrative parent of {obj}"
    ],
    "position baseball_roster_position baseball roster sports_team sports ": [
        "team {obj} has the baseball roster position {subj} ",
        "{subj} is a baseball roster position present in {obj} "
    ],
    "produced_by film ": [
        "the film {obj} was produced by {subj}",
        "{subj} had produced the film {obj} "
    ],
    "film_format film ": [
        "The film {obj} is made in the format {subj}",
        "{subj} is the film format of {obj}"
    ],
    "producer_type tv_producer_term programs_produced tv_producer tv ": [
        "{obj} was the following type of tv producer {subj}",
        "{subj} was the tv producer type of {obj}"
    ],
    "film_release_distribution_medium film_regional_release_date release_date_s film ": [
        "{obj} was released on {subj}",
        "{subj} is the support for {obj}"
    ],
    "film_sets_designed film_set_designer film ": [
        "{obj} designed the film set of {subj} ",
        "{subj} film set were designed by {obj} "
    ],
    "participant popstra base canoodled celebrity popstra base ": [
        "{obj} and {subj} ones canoodled each others",
        "{subj} and {obj} ones canoodled each others"
    ],
    "country_of_origin tv_program tv ": [
        "tv program {obj} originates from {subj}",
        "{subj} is the country from which the tv program {obj} originates"
    ],
    "artist record_label music ": [
        "{obj} is the music record label of {subj}",
        "{subj} has {obj} as its record label"
    ],
    "sport sports_team sports ": [
        "{obj} is a {subj} sports team",
        "{subj} has professional sports team like {obj} "
    ],
    "contact_category phone_sandbox schemastaging base phone_number organization_extra schemastaging base ": [
        "{obj} have its organization phone sandbox in {subj}",
        "{subj} is the contact category phone sandbox of the organization {obj}"
    ],
    "sibling sibling_relationship sibling_s person people ": [
        "{obj} is a sibling of {subj} ",
        "{subj} is a sibling of {obj} "
    ],
    "organization organization_membership member_of organization_member organization ": [
        "{obj} is a memeber of the organisation of the {subj}",
        "{subj} is an organization with member like {obj}"
    ],
    "administrative_area_type administrative_area schema aareas base ": [
        "{obj} has the administrative area type {subj}",
        "{subj} is the administrative area type of {obj}"
    ],
    "actor regular_tv_appearance regular_cast tv_program tv ": [
        "One of the cast of {obj} was {subj}",
        "The actor {subj} made regular tv appearance in the program {obj}"
    ],
    "legislative_sessions government_position_held members legislative_session government ": [
        "Members of the legislative branch convened during the respective legislative sessions of both {obj} and {subj}",
        "Members of the legislative branch convened during the respective legislative sessions of both {obj} and {subj}"
    ],
    "special_performance_type performance actor film ": [
        "actor {obj} has the special performance {subj}",
        "special performance {subj} was performed by the actor {obj}"
    ],
    "school sports_league_draft_pick draft_picks professional_sports_team sports ": [
        "{obj} team selected players from {subj} during the draft picks process ",
        "during the draft picks process {subj} has players picked up for {obj} team"
    ],
    "category webpage topic common ": [
        "{obj} has the category webpage topic common {subj}",
        "{subj} is the category webpage topic common {obj}"
    ],
    "school_type educational_institution education ": [
        "{obj} is the following type of university {subj}",
        "{subj} is the type of university like {obj}"
    ],
    "film_festivals film ": [
        "{obj} was screened at {subj}",
        "during {subj} festival, {obj} was screened"
    ],
    "participant popstra base friendship celebrity popstra base ": [
        "{obj} and {subj} share a friendship thanks to their pop culture implication ",
        "{subj} and {obj} share a friendship thanks to their pop culture implication "
    ],
    "symptom_of symptom medicine ": [
        "{obj} is a symptom of {subj}",
        "{subj} causes symptoms like {obj}"
    ],
    "cause_of_death people ": [
        "{obj} was the cause of death of {subj}",
        "{subj} died of a {obj}"
    ],
    "currency dated_money_value measurement_unit local_tuition university education ": [
        "The local tuition of the the university {obj} is in the currency {subj}",
        "{subj} is the currency used to pay the local tuition for the university of {obj}"
    ],
    "film_art_direction_by film ": [
        "the art director of the film {obj} was {subj}",
        "{subj} was the art director of the film {obj}"
    ],
    "split_to gardening_hint dataworld ": [
        "{obj} split to gardening hint dataworld {subj} ",
        "{subj} split to gardening hint dataworld {obj} "
    ],
    "citytown mailing_address location headquarters organization ": [
        "{obj} headquarters are located in {subj}",
        "{subj} is the mailing address location of the headquarters of the organization {obj} "
    ],
    "director film ": [
        "{obj} was the director of the film {subj}",
        "{subj} was directed by {obj}"
    ],
    "person personal_film_appearance personal_appearances film ": [
        "the film {obj} hosted the {subj} ",
        "{subj} made an appearance in the film {obj} "
    ],
    "student students_graduates educational_institution education ": [
        "{obj} was the university of {subj}",
        "{subj} was ones a student of {obj}"
    ],
    "legislative_sessions government_position_held government_positions_held politician government ": [
        "{obj} held a government position participating in legislative sessions in {subj}",
        "During {subj} , the politician {obj} participated to legislative sessions"
    ],
    "artists genre music ": [
        "{obj} is the music genre of the artist {subj} ",
        "{subj} is an artist playing {obj} music genre"
    ],
    "company employment_tenure people_with_this_title job_title business ": [
        "{obj} is a job title in {subj}",
        "{subj} employs people with the following title job {obj}"
    ],
    "production_companies film ": [
        "{obj} was produced by {subj} company",
        "{subj} is the production company of the film {obj}"
    ],
    "award_honor awards_won award_winner award ": [
        "{obj} and {subj} are award winners",
        "{subj} and {obj} are award winner"
    ],
    "student students_majoring field_of_study education ": [
        "{obj} was the major field of study of {subj}",
        "{subj} majored in the field {obj}"
    ],
    "participating_countries olympic_games olympics ": [
        "during {obj} the country {subj} participated",
        "{subj} was one of the countries which participated to {obj}"
    ],
    "language film ": [
        "{obj} is a film in {subj} language",
        "{subj} is the language of the film {obj}"
    ],
    "country mailing_address location headquarters organization ": [
        "{obj} has its country mailing address located in {subj}",
        "{subj} is the mailing address location of {obj} country"
    ],
    "tv_program tv_program_writer_relationship tv_programs tv_writer tv ": [
        "{obj} is the tv program writer of {subj} ",
        "{subj} was written by {obj}"
    ],
    "program tv_producer_term programs_produced tv_producer tv ": [
        "{obj} was the tv producer of {subj}",
        "{subj} was produced by {obj}"
    ],
    "sports olympic_games olympics ": [
        "{obj} had {subj} as one of its olympic sports ",
        "{subj} was an olympic sport for the {obj}"
    ],
    "place_of_burial deceased_person people ": [
        "{obj} is buried in {subj}",
        "{subj} is the burial place of {obj}"
    ],
    "colors educational_institution education ": [
        "the color of {obj} of the educational institution {subj}",
        "{subj} is the color of the educational institution {obj}"
    ],
    "official_language country location ": [
        "{obj} official language is {subj}",
        "{subj} is the official language in {obj}"
    ],
    "taxonomy taxonomy_entry random tsegaran user entry taxonomy_object random tsegaran user ": [
        "{obj} falls withing the taxonomy of the {subj}",
        "{subj} has {obj} within its taxonomie"
    ],
    "currency dated_money_value measurement_unit gdp_nominal_per_capita statistical_region location ": [
        "{obj} uses the {subj} as a currency reference point for measuring the nominal GDP",
        "{subj} is the currency used as a reference to measure the nominal GDP of {obj}"
    ],
    "position sports_team_roster sports current_roster football_team american_football ": [
        "{obj} has a {subj} in their current roster",
        "{subj} is one of the roster of {obj} "
    ],
    "jurisdiction_of_office government_position_held government_positions_held politician government ": [
        "the politician {obj} was under the jurisdiction of {subj}",
        "{subj} had the politician {obj}"
    ],
    "service_location phone_sandbox schemastaging base phone_number organization_extra schemastaging base ": [
        "{obj} has it service base located in {subj}",
        "{subj} is the location where {obj} has its service based"
    ],
    "person tv_regular_personal_appearance tv_regular_personal_appearances non_character_role tv ": [
        "{subj} made regular tv appearance as a {obj} character ",
        "{subj} made regular tv appearance as a {obj} character "
    ],
    "jurisdiction_of_office government_position_held officeholders government_office_category government ": [
        "{obj} is the government office category of the position held by officeholders in {subj} ",
        "in the jurisdiction of {subj} the government office position is {obj}"
    ],
    "service_language phone_sandbox schemastaging base phone_number organization_extra schemastaging base ": [
        "{obj} has its base phone number in {subj}",
        "{subj} is the base phone number of {obj}"
    ],
    "place_of_death deceased_person people ": [
        "{obj} died in {subj}",
        "{subj} was the place of death of {obj}"
    ],
    "position football_roster_position current_roster football_team soccer ": [
        "in the current roster of the {obj} football team, {subj} is one of the position",
        "{subj} is one of the position in the roster of {obj} football team"
    ],
    "friend friendship celebrity_friends celebrity celebrities ": [
        "{obj} and {subj} are two celebrity friends",
        "{subj} and {obj} are two celibrity friends"
    ],
    "exported_to imports_and_exports places_exported_to statistical_region location ": [
        "{obj} imports goods from {subj} ",
        "{subj} imports goods from {obj}"
    ],
    "capital country location ": [
        "{obj} is the country to which {subj} is the capital ",
        "{subj} is the capital of {obj}"
    ],
    "major_field_of_study students_majoring field_of_study education ": [
        "{obj} is the major field of study for {subj}",
        "{subj} is a sub field of study of {obj}"
    ],
    "diet practicer_of_diet eating base ": [
        "{obj} has a diet based on {subj}",
        "{subj} is the diet eating base of {obj} "
    ],
    "contains location ": [
        "{obj} contains the location {subj} ",
        "{subj} is a place of {obj}"
    ],
    "registering_agency non_profit_registration registered_with non_profit_organization organization ": [
        "{obj} is register with {subj} agency as a non profit",
        "{subj} agency has register {obj} as a non profit"
    ],
    "instrumentalists instrument music ": [
        "{obj} was the instrument of {subj}",
        "{subj} was a instrumentalists of {obj}"
    ],
    "medal olympic_medal_honor medals_won olympic_participating_country olympics ": [
        "{obj} won the {subj} during the olympic games",
        "{subj} was won by {obj} during the olympics games"
    ],
    "state bibs_location biblioness base ": [
        "{obj} is the state location of the biblioness base for {subj}",
        "{subj} biblioness base has its state location in {obj}"
    ],
    "film_production_design_by film ": [
        "In the film {obj} the production design was done by {subj} ",
        "{subj} was the production designer in {obj} "
    ],
    "time_zones location ": [
        "{obj} follow the time zone location {subj} ",
        "{subj} is the time zone of {obj}"
    ],
    "disciplines_or_objects award_category award ": [
        "{obj} is a award for {subj}",
        "{subj} are object to the following award {obj}"
    ],
    "month travel_destination_monthly_climate climate travel_destination travel ": [
        "{obj} is a good destination to travel in {subj} ",
        "{subj} is the best month to travel in {obj} "
    ],
    "notable_people_with_this_condition disease medicine ": [
        "{obj} is a condition disease which affected famous person {subj}",
        "{subj} was a famous person with {obj}"
    ],
    "adjoins adjoining_relationship adjoin_s location ": [
        "{obj} adjoins {subj} ",
        "{subj} adjoins {obj}"
    ],
    "program_creator tv_program tv ": [
        "tv program {obj} was created by {subj}",
        "{subj} was the creator of {obj}"
    ],
    "cinematography film ": [
        "In the film {obj}, the cinematography was handled by {subj}",
        "{subj} handled the cinematography for the film {obj}  "
    ],
    "currency dated_money_value measurement_unit net_worth person_extra schemastaging base ": [
        "{obj}'s net worth to gauge financial success is measured in {subj}",
        "{subj} is the currency used to gauge financial success of {obj}"
    ],
    "category_of award_category award ": [
        "{obj} is a category of the award {subj}",
        "{subj} has the category {obj}"
    ],
    "award_nomination award_nominations award_nominee award ": [
        "Both {obj} and {subj} have been recognized as nominees",
        "Both {subj} and {obj} have been recognized as nominees"
    ],
    "educational_institution educational_institution_campus education ": [
        "{obj} is an educational institution with its campus located in {subj}",
        "{subj} is the campus location of the educational institution {obj}"
    ],
    "costume_design_by film ": [
        "costumes in {obj} were designed by {subj}",
        "{subj} designed the costumes in {obj}"
    ],
    "team sports_league_participation teams sports_league sports ": [
        "{obj} is a sport team group to which {subj} team belongs",
        "the {subj} participates to the sport team league {obj}"
    ],
    "student people_with_this_degree educational_degree education ": [
        "{obj} is the type of degree {subj}",
        "{subj} has a degree in {obj}"
    ],
    "languages person people ": [
        "{obj} speaks {subj}",
        "{subj} was the language {obj} spokes"
    ],
    "program tv_network_duration programs tv_network tv ": [
        "{obj} broadcasted the tv program {subj}",
        "{subj} was broadcasted by {obj} "
    ],
    "colors sports_team sports ": [
        "{obj} color sport team is {subj}",
        "{subj} is the color of the sport team {obj}"
    ],
    "languages tv_program tv ": [
        "TV program {obj} was aired in {subj}",
        "{subj} was the language of the TV program {obj}"
    ],
    "family instrument music ": [
        "{obj} belongs to the instrument family of {subj}",
        "{subj} is the family instrument of {obj}"
    ],
    "currency dated_money_value measurement_unit international_tuition university education ": [
        "{obj} has its international tuition in currency {subj}",
        "{subj} is the currency for the international tuition fees in {obj}"
    ],
    "place_founded organization ": [
        "{obj} was founded in the place {subj}",
        "{subj} is the place where {obj} was founded "
    ],
    "olympics olympic_athlete_affiliation athletes olympic_participating_country olympics ": [
        "{obj} participated at the {subj}",
        "during {subj} the country {obj} participated"
    ],
    "capital administrative_area schema aareas base ": [
        "{obj} has its administrative capital in {subj}",
        "{subj} is the capital of the administrative area of {obj}"
    ],
    "crewmember film_crew_gig other_crew film ": [
        "{obj} had {subj} within its crew memebers",
        "{subj} was one of the crew memeber for the film {obj} "
    ],
    "recording_contribution guest_performances performance_role music ": [
        "{obj} recording contribution guest performances performance role music {subj}",
        "{subj} recording contribution guest performances performance role music {obj}"
    ],
    "producer_type tv_producer_term tv_producer tv_program tv ": [
        "{obj} is tv producer of the following type {subj} ",
        "{subj} is the type of tv producer of {obj}"
    ],
    "award award_honor awards_won award_winning_work award ": [
        "{obj} won the award {subj}",
        "{subj} is an award won by the {obj} "
    ],
    "performance_role recording_contribution contribution artist music ": [
        "{obj} has the performance role recording contribution of {subj}",
        "{subj} was the recording contribution role of {obj} "
    ],
    "music film ": [
        "the music film {obj} was composed by {subj}",
        "{subj} participated in the music film {obj}"
    ],
    "celebrities_impersonated celebrity_impressionist americancomedy base ": [
        "{obj} has impersonated {subj} ",
        "{subj} was impersonated by {obj}"
    ],
    "language dubbing_performance dubbing_performances actor film ": [
        "{obj} is a dubbing actor in {subj}",
        "{subj} is the dubbing language performed by the actor {obj}"
    ],
    "risk_factors disease medicine ": [
        "{obj} risks is increased by the following factor {subj}",
        "{subj} is a risk factor of {obj}"
    ],
    "currency dated_money_value measurement_unit estimated_budget film ": [
        "the budget of {obj} was estimated in {subj}",
        "{subj} was the currency used to estimate the budget of {obj}"
    ],
    "film film_film_distributor_relationship films_distributed film_distributor film ": [
        "{obj} is the film ditributor of {subj}",
        "The film {subj} was distributed by {obj}"
    ],
    "country film ": [
        "{obj} is the country of the film {subj}",
        "{subj} is the country of the film {obj}"
    ],
    "award award_nomination award_nominations award_nominee award ": [
        "{obj} had an award nomination at the award {subj}",
        "{subj} was the award were {obj} was a nominee"
    ],
    "team ncaa_tournament_seed marchmadness base seeds ncaa_basketball_tournament marchmadness base ": [
        "{obj} was a basketball tornament where {subj} participated as a seeded team",
        "{subj} basketball team participated in the {obj} as a seeded team "
    ],
    "romantic_relationship sexual_relationships celebrity celebrities ": [
        "{obj} was involved in a romantic and sexual relationship with the celebrity {subj}",
        "{subj} was involved in a romantic and sexual relationship with the celebrity {obj}"
    ],
    "gender person people ": [
        "{obj} has the gender of a {subj}",
        "{subj} is the gender of {obj}"
    ],
    "position sports_team_roster sports current_roster football_team soccer ": [
        "The team {obj} has the following position {subj} ",
        "{subj} is a football position of the {obj}"
    ],
    "ceremony award_honor winners award_category award ": [
        "{obj} is an award cathegory of {subj}",
        "{subj} is the ceremony award for the {obj}"
    ],
    "state_province_region mailing_address location headquarters organization ": [
        "{obj} has its headquarter mailing address in the state province of {subj}",
        "{subj} is the state province mailing address of {obj}"
    ],
    "adjustment_currency adjusted_money_value measurement_unit gdp_real statistical_region location ": [
        "{obj} uses {subj} as a reference for currency adjustments",
        "{subj} is the currency used in {obj} for its own currency adjustment"
    ],
    "type_of_appearance personal_film_appearance films person_or_entity_appearing_in_film film ": [
        "{obj} had made appearance in {subj}",
        "{subj} is the type of public appearance {obj} made "
    ],
    "ethnicity people ": [
        "{obj} is the ethnicity of {subj}",
        "{subj} belong to the ethnicity of {obj}"
    ],
    "program tv_regular_personal_appearance tv_regular_appearances tv_personality tv ": [
        "{obj} made regular tv appearance in the show {subj} ",
        "{subj} frequently hosted the celebrity {obj} "
    ],
    "film_release_region film_regional_release_date release_date_s film ": [
        "the region where the film {obj} was released is {subj}",
        "{subj} is a region where the film {obj} was released"
    ],
    "position basketball_roster_position basketball roster sports_team sports ": [
        "the basketball team {obj} has the roster {subj}",
        "{subj} is a roster in the basketball team {obj}"
    ],
    "season baseball_team_stats team_stats baseball_team baseball ": [
        "{obj} baseball team made major team stats during the baseball season {subj} ",
        "during the baseball season {subj} the baseball team {obj} made major team stats "
    ],
    "titles netflix_genre media_common ": [
        "{obj} is the media genre of {subj}",
        "the netflix show {subj} is a {obj}"
    ],
    "genre tv_program tv ": [
        "The tv program {obj} has the following genre {subj}",
        "{subj} is the genre of the tv program {obj} "
    ],
    "profession person people ": [
        "{obj} has the profession of {subj}",
        "{subj} is the profession of {obj}"
    ],
    "place hud_county_place location ": [
        "{obj} is a place hud county location of {subj}",
        "{obj} is a place hud county location of {subj}"
    ],
    "basic_title government_position_held government_positions_held politician government ": [
        "{obj} has the governement position of {subj}",
        "As a member of {subj} , {obj} held various governement positions"
    ],
    "nutrient nutrition_fact nutrients food ": [
        "{obj} contains the nutrient {subj}",
        "{subj} is a nutrient in {obj} "
    ],
    "currency dated_money_value measurement_unit gdp_nominal statistical_region location ": [
        "{obj} gdp is measured in {subj} ",
        "{subj} is the currency used to measure the gdp of {obj}"
    ],
    "legislative_sessions government_position_held members governmental_body government ": [
        "{obj} was the governemental body during {subj}",
        "{obj} was the governemental body during {subj}"
    ],
    "prequel film ": [
        "the film {obj} has the prequel {subj}",
        "{subj} is the prequel of {obj}"
    ],
    "currency dated_money_value measurement_unit endowment endowed_organization organization ": [
        "{obj} is paid in {subj}",
        "{subj} is the currency used to pay {obj}"
    ],
    "county hud_county_place location ": [
        "{obj} is located in the county of {subj} ",
        "{subj} is the county were {obj} is located"
    ],
    "origin artist music ": [
        "The music artist {obj} originated from {subj} ",
        "{subj} is the origin location of the artist {obj}"
    ],
    "religion religion_percentage religions statistical_region location ": [
        "{obj} has an important religious percentage of {subj}",
        "{subj} is significative religions distribution in the region of {obj} "
    ],
    "team sports_team_roster teams pro_athlete sports ": [
        "{obj} was a pro athlete in {subj}",
        "{subj} was the team of {obj}"
    ],
    "film performance film_performance_type special_film_performance_type film ": [
        "{obj} is the type of actor performance for the film {subj}",
        "the film {subj} has the type of actor performance {obj}"
    ],
    "draft sports_league_draft_pick draft_picks professional_sports_team sports ": [
        "New player for baseball league {obj} were selected during {subj}",
        "During {subj} the sport league {obj} selected new players"
    ],
    "countries_spoken_in human_language language ": [
        "{obj} is spoken in {subj}",
        "{subj} spokes {obj}"
    ],
    "source dated_integer measurement_unit estimated_number_of_mortgages hud_foreclosure_area location ": [
        "In {obj} area the source used to estimate the number of mortgages hub foreclosure is {subj} ",
        "{subj} is the source for the measurements and estimations of the number of mortgages hub foreclosure in {obj}"
    ],
    "currency dated_money_value measurement_unit revenue business_operation business ": [
        "{obj} makes its revenue in {subj} ",
        "{subj} is the currency used to evaluate the revenues of {obj}"
    ],
    "peer_relationship peers influence_node influence ": [
        "{obj} and {subj} shared a peer relationship",
        "{subj} and {obj} shared a peer relationship"
    ],
    "instance_of_recurring_event event time ": [
        "{obj} is a recurring event of {subj}",
        "{subj} is the main event of {obj}"
    ],
    "currency dated_money_value measurement_unit rent50_2 statistical_region location ": [
        "{obj} uses the currency {subj} ",
        "{subj} is the currency used in {obj}"
    ],
    "spouse marriage spouse_s person people ": [
        "{obj} is the spouse of {subj}",
        "{subj} has his / her spouse named {obj}"
    ],
    "religion person people ": [
        "{obj} practice {subj} religion",
        "{subj} is the religion of {obj}"
    ],
    "featured_film_locations film ": [
        "The film {obj} featured {subj} location ",
        "{subj} was the location featured in {obj} "
    ],
    "seasonal_months produce_availability localfood base produce_available seasonal_month localfood base ": [
        "{obj} and {subj} are both seasonal months to produce local food ",
        "{obj} and {subj} are both seasonal months to produce local food "
    ],
    "group group_membership membership group_member music ": [
        "{obj} is a memeber of the music group {subj}",
        "{subj} is the music group of {obj}"
    ],
    "mode_of_transportation transportation how_to_get_here travel_destination travel ": [
        "to get in {obj} the transportation mode is {subj}",
        "{subj} is the transportation mode to get in {obj}"
    ],
    "story_by film ": [
        "The film {obj} was written by {subj}",
        "{subj} wrote the film {obj}"
    ],
    "entity_involved event culturalevent base ": [
        "the cultural event {obj} involved {subj}",
        "{subj} were involved in the cultural event {obj} "
    ],
    "written_by film ": [
        "the film {obj} was written by {subj}",
        "{subj} wrote the movie {obj}"
    ],
    "fraternities_and_sororities university education ": [
        "{obj} hosts the sorority {subj}",
        "{subj} is a sorority of {obj}"
    ],
    "second_level_divisions country location ": [
        "{obj} has second level division such as {subj} ",
        "{subj} belongs to the second level country division of the {obj}"
    ],
    "athlete pro_sports_played pro_athletes sport sports ": [
        "{obj} had pro athlete like {subj} ",
        "{subj} was a pro athlete in {obj}"
    ],
    "nominated_for award_nomination award_nominations award_nominee award ": [
        "{obj} received an award nomination for the movie {subj}",
        "the movie {subj} had {obj} nominated for an award"
    ],
    "country olympic_athlete_affiliation athletes olympic_sport olympics ": [
        "{obj} olympic athlete can be affiliated to the country {subj} ",
        "{subj} country has olympic athlete {obj}"
    ],
    "award_winner award_honor winners award_category award ": [
        "{obj} is an award category won by {subj} ",
        "{subj} won an award in the category {obj} "
    ],
    "country bibs_location biblioness base ": [
        "The biblioness of {obj} is located in the country {subj}",
        "{subj} is the country of the biblioness of {obj}"
    ],
    "honored_for award_honor awards_won award_winning_work award ": [
        "{obj} and {subj} were both honoured by an award",
        "{obj} and {subj} were both honoured by an award"
    ],
    "place_of_birth person people ": [
        "{obj} was born in the place {subj} ",
        "{subj} is the place of birth of {obj}"
    ],
    "interests philosopher philosophy alexander user ": [
        "philosopher {obj} had interests for {subj}",
        "{subj} was one of the interests of the philosopher {obj}"
    ],
    "teams sports_team_location sports ": [
        "The location {obj} host the sports team {subj} ",
        "The team {subj} is located in {obj}"
    ],
    "role track_contribution track_contributions artist music ": [
        "{obj} had a role trak in as a music artic with a contribution track with {subj}",
        "{subj} was an contribution of {obj} in music "
    ],
    "vacationer vacation_choice popstra base vacationers location popstra base ": [
        "{obj} is the vacation location base of the star {subj}",
        "The star {subj} spends his or her vacation in the location {obj}"
    ],
    "politician political_party_tenure politicians_in_this_party political_party government ": [
        "{obj} was the political party of {subj}",
        "{subj} is a politician from the party {obj}"
    ],
    "administrative_division administrative_division_capital_relationship capital_of capital_of_administrative_division location ": [
        "{obj} serves as the administrative division within {subj}",
        "{subj} as its administrative division in {obj} "
    ],
    "nationality person people ": [
        "the nationality of {obj} is {subj} ",
        "{subj} is the nationality of {obj}"
    ],
    "artist content broadcast ": [
        "{obj} broadcast the artist {subj}",
        "{subj} is broadcasted on {obj}"
    ],
    "member_states international_organization default_domain ktrueman user ": [
        "{obj} international organization has country memebers like {subj}",
        "{subj} is a country member of the {obj}"
    ],
    "form_of_government country location ": [
        "{obj} has the government form of a {subj} ",
        "{subj} is the government form of {obj}"
    ],
    "performance actor film ": [
        "{obj} made an apparition in the film {subj}",
        "the film {subj} ones hosted {obj} as an actor"
    ],
    "award_winner award_honor awards_won award_winning_work award ": [
        "the work {obj} of {subj} helped him or her to win an award",
        "{subj} won an award for his or her work in {obj} "
    ],
    "first_level_division_of administrative_division location ": [
        "{obj} is the first level administrative division within {subj} ",
        "{subj} has first level administrative division like {obj}"
    ],
    "actor dubbing_performance dubbing_performances film ": [
        "{obj} was dubbed by {subj}",
        "{subj} was a dubbing actor in {obj}"
    ],
    "currency dated_money_value measurement_unit gni_per_capita_in_ppp_dollars statistical_region location ": [
        "the location {obj} measure its unit gmi per capita in ppp using the {subj} currency",
        "{subj} currency is used to measure the units gmi per capita in ppp in the location {obj}"
    ],
    "executive_produced_by film ": [
        "The executive producer of the film {obj} was {subj}",
        "{subj} was the executive producer of {obj}"
    ],
    "honored_for award_honor awards_presented award_ceremony award ": [
        "during the award ceremony of {obj}, {subj} was honored",
        "{subj} was honored during the award ceremony {obj} "
    ],
    "locations event time ": [
        "{obj} took place in the location {subj} ",
        "{obj} as one of the location were {subj} took place"
    ],
    "countries_within continents locations base ": [
        "{obj} is the continent of {subj}",
        "{subj} is a country of the continent {obj}"
    ],
    "award_winner award_honor awards_presented award_ceremony award ": [
        "During the {obj} the artist {subj} won an award",
        "{subj} won a award at {obj}"
    ],
    "major_field_of_study people_with_this_degree educational_degree education ": [
        "{obj} is the major field for a degree in {subj} ",
        "A degree in {subj} has a major in {obj}"
    ],
    "dog_breed dog_city_relationship petbreeds base top_breeds city_with_dogs petbreeds base ": [
        "{obj} mostly have the pet bread {subj}",
        "The dog bread {subj} is the top breed of {obj} "
    ],
    "role group_membership regular_performances performance_role music ": [
        "{obj} can perform the music role of {subj}",
        "{subj} sound can be performed by {obj}"
    ],
    "participant popstra base dated celebrity popstra base ": [
        "{obj} and {subj} are participant in the popstart culture and dated",
        "{obj} and {subj} are participant in the popstart culture and dated"
    ],
    "school sports_league_draft_pick picks sports_league_draft sports ": [
        "during {obj}, {subj} produced draft picks selected by teams in the sports league",
        "during {obj}, {subj} produced draft picks selected by teams in the sports league"
    ],
    "role group_membership membership group_member music ": [
        "{obj} had the role of the {subj} in his music group ",
        "{subj} is the role of the member {obj} in his music group"
    ],
    "geographic_distribution ethnicity people ": [
        "{obj} are a ethinicity present in {subj}",
        "{subj} is a country with a disaspora of {obj}"
    ],
    "location_of_ceremony marriage unions_of_this_type marriage_union_type people ": [
        "{obj} ceremony takes place in {subj}",
        "In {subj} there are {obj} ceremonies "
    ],
    "sports olympic_games default_domain jg user ": [
        "During {obj} there was the sports olympic games {subj} ",
        "{subj} was one of the olympic sports during {obj}"
    ],
    "organization_relationship child organization ": [
        "{obj} is the parent organisation of {subj}",
        "{subj} is the child organisation of {obj}"
    ],
    "country administrative_division location ": [
        "{obj} has its county administrative division in  {subj}",
        "{subj} is the county administrative of the division of {obj} "
    ],
    "position sports_team_roster sports current_roster hockey_team ice_hockey ": [
        "the hockey team {obj} have {subj} i, their their roster",
        "{subj} is present in {obj} hockey team roster"
    ],
    "district_represented government_position_held members legislative_session government ": [
        "{obj} has legislative member from {subj}",
        "{subj} district had a member who had legislative position in {obj}"
    ],
    "current_club x2010fifaworldcupsouthafrica base current_world_cup_squad world_cup_squad x2010fifaworldcupsouthafrica base ": [
        "{obj} includes players from {subj} in its current World Cup squad for the 2010 FIFA World Cup in South Africa.",
        "{subj} player are included in the {obj} for the World Cup for the 2010 FIFA World Cup in South Africa."
    ],
    "team sports_team_roster sports current_team football_player soccer ": [
        "{obj} was a football soccer player in the team {subj}",
        "{subj} was the football team of the soccer player {obj} "
    ],
    "partially_contains location ": [
        "{obj} partially contains {subj}",
        "{subj} is partially contained in {obj}"
    ],
    "position football_roster_position american_football roster sports_team sports ": [
        "football team {obj} had a {subj} position in its roster",
        "{subj} is a position is the football roster of the team {obj}"
    ],
    "organizations_founded organization_founder organization ": [
        "{obj} founded the organisation {subj}",
        "{subj} was founded by {obj}"
    ],
    "position sports_team_roster players sports_position sports ": [
        "{obj} and {subj} are both roster player positions in sports teams",
        "{obj} and {subj} are both roster player positions in sports teams"
    ],
    "nominated_for award_nomination award_nominations award_nominated_work award ": [
        "{obj} and {subj} were both nominated for an award for their work",
        "{subj} and {obj} were both nominated for an award for their work"
    ],
    "combatants military_combatant_group military_conflicts military_combatant military ": [
        "Combatant from {obj} were opposed to {subj} in a military conflict",
        "Combatant from {subj} were opposed to {obj} in a military conflict"
    ],
    "county_seat us_county location ": [
        "County seat of {obj} are located in {subj}",
        "{subj} are the location for the {obj} seat"
    ],
    "industry business_operation business ": [
        "{obj} has its industry business operations in {subj}",
        "{subj} is the industry business operation of {obj}"
    ],
    "team sports_team_roster players sports_position sports ": [
        "{obj} is a postion in the team sport roster of tge {subj}",
        "sport team {subj} has the position {obj} in its roster"
    ],
    "inductee hall_of_fame_induction inductees hall_of_fame award ": [
        "{obj} is the hall of fame for {subj}",
        "{subj} is in the hall of fame {obj}"
    ],
    "parent_genre genre music ": [
        "{obj} is a musical sub genre of {subj}",
        "{subj} is the parent genre music of {obj}"
    ],
    "influenced_by influence_node influence ": [
        "{obj} was influence by the writter {subj}",
        "{subj} influence the writter {obj}"
    ],
    "film_crew_role film_crew_gig other_crew film ": [
        "The film {obj} has a crew of {subj}",
        "A crew of {subj} was present for the film {obj}"
    ],
    "olympics olympic_athlete_affiliation athletes olympic_sport olympics ": [
        "the olympic sport {obj} had athletes affiliated to {subj}",
        "during {subj} , athletes from {obj} were affiliated"
    ],
    "currency dated_money_value measurement_unit domestic_tuition university education ": [
        "{obj} has it tuiotion fees measured in {subj}",
        "{subj} is the currency unit measurement for the tuition fees of {obj}"
    ],
    "specialization_of profession people ": [
        "{obj} is a professional specialisation of {subj}",
        "the profession {subj} can specialise in {obj}"
    ],
    "olympics olympic_medal_honor medals_won olympic_participating_country olympics ": [
        "{obj} won the olympic participating honor medal during the {subj} ",
        "{subj} was when {obj} recived the participating honor medal"
    ],
    "type_of_union marriage spouse_s person people ": [
        "{obj} is united to his or her spouse through a {subj} ",
        "{subj} is the type of union between {obj} and his or her spouse"
    ],
    "major_field_of_study students_graduates educational_institution education ": [
        "{obj} gives a major in {subj}",
        "{subj} is a major given in {obj}"
    ],
    "list ranking appears_in_ranked_lists ranked_item award ": [
        "{obj} appeared in the ranking ward {subj} ",
        "{obj} ranks {subj} "
    ],
    "film_release_region film_cut runtime film ": [
        "{obj} was released in {subj} and had its runtime determined by the film cut",
        "{subj} was the released region for the film {obj} which had its runtime determined by the film cut"
    ],
    "position_s football_historical_roster_position american_football roster sports_team sports ": [
        "in the historical roster of {obj} football team, {subj} has been an important position",
        "{subj} has been an important position in the historical roster of {obj} football team"
    ],
    "role track_contribution track_performances performance_role music ": [
        "{obj} often play with {subj} in music performances",
        "{subj} often play with {obj} in music performances"
    ],
    "campuses educational_institution education ": [
        "{obj} campus location of {subj}",
        "{subj} is the campus location of {obj}"
    ],
    "film_regional_debut_venue film_regional_release_date release_date_s film ": [
        "film {obj} mad its regional debut at {subj}",
        "{subj} was the festival were the film {obj} made its regional debut "
    ],
    "group group_membership regular_performances performance_role music ": [
        "{obj} is a performance role music present in the group {subj}",
        "the music group {subj} the music role {obj} "
    ],
    "organization leadership leaders role organization ": [
        "{obj} is the leader role for the organisation of {subj}",
        "{subj} as a {obj} as leader"
    ],
    "location_of_ceremony marriage spouse_s person people ": [
        "{obj} made his or her marriage ceremony in {subj}",
        "{subj} was the location of the marriage of {obj}"
    ],
    "genre film ": [
        "The film {obj} belongs to the genre of {subj}",
        "{subj} is the genre of the film {obj} "
    ],
    "participant popstra base breakup celebrity popstra base ": [
        "celebrity {obj} and {subj} broke up",
        "celebrity {subj} and {obj} broke up"
    ],
    "company employment_tenure business employment_history person people ": [
        "{obj} was employed by {subj}",
        "{subj} employed {obj}"
    ],
    "languages_spoken ethnicity people ": [
        "ethnic minority {obj} speaks {subj}",
        "{subj} is spoken by the ethnic minority {obj}"
    ],
    "region film_film_distributor_relationship distributors film ": [
        "{obj} was distributed by a distributor from {subj}",
        "{subj} is the country from which {obj} distributor comes from"
    ],
    "institution people_with_this_degree educational_degree education ": [
        "{obj} degree can be delivered by {subj} educational institution",
        "{subj} educational institution delivers degrees like {obj} "
    ],
    "currency dated_money_value measurement_unit operating_income business_operation business ": [
        "{obj} measures its operatinf income in {subj} currency",
        "{subj} is the currency used by {obj} to measure its operating income"
    ]
}