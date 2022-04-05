import time

from torch import torch
from torch.utils.data import DataLoader, random_split

from model import collate_batch, dataset, device, TextClassificationModel, text_pipeline, save_model, load_model

training_data = dataset
vocab = training_data.vocab_list.get('combined_description')

num_class = len(set([label for (text, label) in training_data]))
vocab_size = len(vocab)
emsize = 128

tc_model = load_model()
if not tc_model:
    tc_model = TextClassificationModel(vocab_size, emsize, num_class).to(device)
else:
    print('model loaded')


def train(dataloader, model):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500
    start_time = time.time()

    for idx, (label, text, offsets) in enumerate(dataloader):
        optimizer.zero_grad()
        predicted_label = model(text, offsets)
        loss = criterion(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader),
                                              total_acc / total_count), elapsed)
            total_acc, total_count = 0, 0
            start_time = time.time()


def evaluate(dataloader, model):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            predicted_label = model(text, offsets)
            criterion(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc / total_count


# Hyperparameters
EPOCHS = 10  # epoch
LR = 5  # learning rate
BATCH_SIZE = 64  # batch size for training

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(tc_model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.1)
total_accu = None
train_dataset, test_dataset = dataset, dataset
num_train = int(len(train_dataset) * 0.95)
split_train_, split_valid_ = random_split(train_dataset, [num_train, len(train_dataset) - num_train])

train_dataloader = DataLoader(split_train_, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_batch)
valid_dataloader = DataLoader(split_valid_, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=True, collate_fn=collate_batch)

for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    train(train_dataloader, tc_model)
    accu_val = evaluate(valid_dataloader, tc_model)
    if total_accu is not None and total_accu > accu_val:
        scheduler.step()
    else:
        total_accu = accu_val
    print('-' * 59)
    print('| end of epoch {:3d} | time: {:5.2f}s | '
          'valid accuracy {:8.3f} '.format(epoch,
                                           time.time() - epoch_start_time,
                                           accu_val))
    print('-' * 59)

save_model(tc_model)


def predict(text, _text_pipeline):
    with torch.no_grad():
        text = torch.tensor(_text_pipeline(text))
        output = tc_model(text, torch.tensor([0]))
        return output.argmax(1).item()


job_label = {0: 'Real', 1: 'Fake'}
tc_model = tc_model.to("cpu")
# F,R,R,F,F
test_post = [
    'aker solution global provider product system service oil gas industry engineering design technology bring discovery production maximize recovery petroleum field employ approximately 28000 people 30 country go url0fa3f7c5e23a16de16a841e368006cae916884407d90b154dfef3976483a71ae information business people value corporate overviewaker solution global provider product system service oil gas industry engineering design technology bring discovery production maximize recovery petroleum field employ approximately 28000 people 30 country go url0fa3f7c5e23a16de16a841e368006cae916884407d90b154dfef3976483a71ae information business people valueswe looking individual prepared take position position within aker solution also position exciting challenge global oil gas industry face futurewe looking lead mechanical engineer join team houston texasthe lead mechanical engineer responsible providing expertise technical leadership organizationresponsibilities tasksâ€¢ performs mechanical calculation technical analysis various custom component review mechanical design equipment ensure specification metâ€¢ prepares present complex technical report equipment data sheet mrqâ€™s tbeâ€™s mrpâ€™s make recommendation critical engineering issuesâ€¢ work certifying agency product development follows absa registrationsâ€¢ lead review project design decision budget schedulingâ€¢ identifies solution achieve company objective ensure team alignedâ€¢ interface directly customer participates preparing bid proposalsâ€¢ ensure process followed correctly continuously identifies opportunity improve efficienciesâ€¢ ensure team member kept current procedure qms changesâ€¢ provides leadership technical guidance mentorship engineer qualification amp personal attributesâ€¢ mechanical engineering degree equivalent requiredâ€¢ 510 year related experience within epc oil amp gas fabrication shop andor engineering environment requiredâ€¢ registration apegga eligibility member requiredâ€¢ must experience different type mechanical equipment including pressure vessel pump heat exchangersâ€¢ familiarity industry code relevant equipment specifically relevant asme api 610â€¢ ability effectively present information respond question manager employee customer general publicâ€¢ proficiency microsoft office applicationsâ€¢ excellent time managementprioritization skill ability work effectively minimal supervision manage multiple conflicting tasksprojects',
    'headquarteredinmilan new york 400 million user around worldbeintoois true engagement platform full set gamification toolsâ€“badges mission leaderboards contest etcâ€“for mobile web application game national brickandmortar online retailersbeintoogives value usersâ€™ engagement distributing currencyâ€“bedollars canbeused premium offer real world benefit bedollars meritocratic global canberedeemedinthe bestore orinthe online store ofbeintooâ€™s retail partner thousand appealing offersinaddition thisbeintoorewards user engagement achievement letting convert bedollars real cash shop partner retailersâ€™ websitesviabeintooâ€™s reward engine developer attain deeper user engagement monetize inside outside apps get paid every bedollar redeemed ausersponsors hand benefit innovative way engaging customer customized defined loyalty program maximizing efficiency ad campaign reach conversion rate high averagedue global presencebeintoois building worldwide network online offline retailer partner developer accept bedollars virtual credit method payment primary responsibility proactively contact new existing client educates proposes secures mobile advertising campaign drive revenue actively closing deal advertiser agency expand number agency actively business beintoo act primary pointofcontact client advertising campaign work rfp response sale amp monetization director create custom plan proposal response agency rfpâ€™s ability manipulate data system excel powerpoint essential lead daytoday management optimization client campaign liase adoperations publishing team milan ensure campaign delivered successfully upsells client future larger advertising opportunity beintoo master maintains vast knowledge client business competition latest industry news trend evangelize mobile advertising community background qualification 2 year experience bachelorâ€™s degree preferably marketing advertising discipline comfortable close deal start finish cold calling creating proposal negotiating io insertion order experienced runningoverseeingmanaging interactive advertising campaign working agency digital medium planning buying team uk solid commitment sale customer service good interpersonal skill initiative followthrough strongquantitative skill extremely organized highly motivated excellent verbalwritten communication skill experience giving presentation client excellent analytical work â€“ essential ensure campaign trafficking optimisation client bilingual candidate preferred french spanish english german italian',
    'kettle independent digital agency based new york city bay area weâ€™re committed making digital â€” people brand â€” believe digital world offer meet eye every online experience potential change life brand even change world carry philosophy every product build need someone based ca role probably cd storytelling skill strong portfoliothis person would eventually replace maria someone excellent job lead team conceptually execution journalism experience pluspresentation skill neededthis person comfortable global travel  nan',
    'nan greeting vam systemsâ€¦vam systemsis business consulting solution servicescompany operation uae qatar bahrain usa australia singapore amp india offer comprehensive list service field infrastructure management cloud service consulting service banking management resource information technology development telecom aviation retail management amp egovernment offeringsvam system currently looking j2ee developer bahrain operation education bachelorâ€™s degree computer science  itj2ee certificationterms conditionsjoining time frame immediate maximum 2 weeksthe selected candidate shall join vam system â€“ bahrain shall deputed one leading organization bahrainshould yoube interested opportunity please send latest resume m word format earliest emailb08cf5e4101b4b7b7594fe3081f94f7f9a0f2d6192fb5d4a1a72ecf03c816e83 call u 91 phonedf058aa8cbf405b99c6fc6459e7085be12bce0e56bcafe7d52281b99685a4a0f skillset required23 year j2ee experienceexperience requirement gatheringexperience rdbms jdbc jboss websphere ejbsoa linux ldap design pattern methodology eg agile waterfallstrong support experience websphere portalshould experience payment systemsdevelopment loadbalanced environmentstrong analytical skillsexcellent written communication skillsresponsibilitiesundertake development work new existing applicationsinvolvesupport formal analysis user requirement regard new existing system provide appropriate design documentationensure developed software robust meet userâ€™s functional requirementsensure developed software undergone unit system testing prior handover testing integration teaminvolve support estimation design impact analysis task delegated teaminvolvesupport proactively environment closely managed project providing regular task update supporting project management processesdesign develop enterprisescale application javaj2ee platform accordance agreed standard proceduressupport maintain developed application line predefined service level',
    'nan fullservice marketing staffing firm serving company ranging fortune 100 new startup organization work job seeker equally broad range light industrial temporary worker executive level candidatesare looking work home opportunity earn 2500 per week online service representative position would perfect set hour  make money every time decide work  work remotely home  get paid weekly  computer internet requirementsall need access internet participatecomputer internet access valid email address good typing skill '
]

for post in test_post:
    print("This is a %s post" % job_label[predict(post, text_pipeline)])
