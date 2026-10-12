# Live recruitment demonstration

The running Python API at `http://127.0.0.1:8012` was exercised by `tools/demo_recruitment.py` using synthetic donor accounts. The response window was 10 seconds to make automatic escalation visible quickly; the normal default is 60 seconds.

| Stage | Result |
|---|---|
| First wave | 10 nearest reachable matching donors invited |
| First responses | 2 donors accepted; 1 more needed |
| Timeout | Unanswered first-wave invitations expired |
| Second wave | Next 10 donors invited without repeating the first 10 |
| Final acceptance | Third donor accepted from wave 2 |
| Recruitment stopped | 3 accepted, 0 pending, 0 further places needed |
| Donation completion | 0 of 3 completed; acceptance reserves a place only |

Accepted donors, ordered by the synthetic road-time estimate:

| Rank | Donor | Wave | ETA |
|---|---|---|---|
| 1 | donor_003 | 1 | 1.0 min |
| 2 | donor_004 | 1 | 1.3 min |
| 3 | donor_013 | 2 | 4.2 min |

`first_batch.json`, `second_batch.json`, `accepted_donors.json`, `request.json`, and `result.json` are captured API responses/results. The PNG files render the actual Flutter recruitment widget using these captured responses; they are not browser screenshots.

The server remains available for inspection. Sign in as `staff_001`, open the three-unit request `REQ_4f7e6b1871434f55928e58dfdbd70640`, and choose **Accepted donors**.

Verification: 308 Python tests passed; 20 regular Flutter tests passed; Flutter analysis reported no issues; web build succeeded. The optional screenshot test also passed when run with the captured API directory and local fonts.

This demonstration used saved synthetic locations and synthetic routing. Invitations were delivered inside the donor portal; no external push or SMS notifications were sent. It does not demonstrate real GPS accuracy or live Google traffic.
