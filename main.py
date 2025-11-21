from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Any
from datetime import datetime
import json
import xml.etree.ElementTree as ET
import uuid


# Exceptions
class BookingError(Exception):
    pass


class NotFoundError(BookingError):
    pass


class SeatUnavailableError(BookingError):
    pass


class ValidationError(BookingError):
    pass


class PaymentError(BookingError):
    pass


# Простейшие сущности
@dataclass
class ContactInfo:
    phone: Optional[str] = None
    email: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ContactInfo":
        return ContactInfo(phone=d.get("phone"), email=d.get("email"))


@dataclass
class User:
    user_id: str
    name: str
    contact: ContactInfo

    def to_dict(self) -> Dict[str, Any]:
        return {"user_id": self.user_id, "name": self.name, "contact": self.contact.to_dict()}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "User":
        return User(user_id=d["user_id"], name=d["name"], contact=ContactInfo.from_dict(d["contact"]))


@dataclass
class Passenger:
    passenger_id: str
    name: str
    passport: Optional[str] = None
    contact: Optional[ContactInfo] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passenger_id": self.passenger_id,
            "name": self.name,
            "passport": self.passport,
            "contact": self.contact.to_dict() if self.contact else None
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Passenger":
        contact = ContactInfo.from_dict(d["contact"]) if d.get("contact") else None
        return Passenger(passenger_id=d["passenger_id"], name=d["name"], passport=d.get("passport"), contact=contact)


# Seat, Transport, Route, Trip
@dataclass
class Seat:
    seat_id: str
    row: Optional[int]
    number: Optional[int]
    class_type: str  # предполагается "economy", "business", "first", "standard"
    is_reserved: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Seat":
        return Seat(seat_id=d["seat_id"], row=d.get("row"), number=d.get("number"), class_type=d.get("class_type", "standard"), is_reserved=d.get("is_reserved", False))


class Transport:
    """Абстрактный транспорт — автобус/поезд/самолет"""
    def __init__(self, transport_id: str, model: str, capacity: int, seats: Optional[List[Seat]] = None) -> None:
        self.transport_id = transport_id
        self.model = model
        self.capacity = capacity
        self.seats: List[Seat] = seats or []
        if not seats:
            for i in range(1, capacity + 1):
                sid = f"{transport_id}-S{i}"
                self.seats.append(Seat(seat_id=sid, row=None, number=i, class_type="standard", is_reserved=False))

    def get_available_seats(self) -> List[Seat]:
        return [s for s in self.seats if not s.is_reserved]

    def find_seat(self, seat_id: str) -> Seat:
        for s in self.seats:
            if s.seat_id == seat_id:
                return s
        raise NotFoundError(f"Seat {seat_id} not found in transport {self.transport_id}")

    def reserve_seat(self, seat_id: str) -> None:
        seat = self.find_seat(seat_id)
        if seat.is_reserved:
            raise SeatUnavailableError(f"Seat {seat_id} already reserved")
        seat.is_reserved = True

    def release_seat(self, seat_id: str) -> None:
        seat = self.find_seat(seat_id)
        seat.is_reserved = False

    def to_dict(self) -> Dict[str, Any]:
        return {"transport_id": self.transport_id, "model": self.model, "capacity": self.capacity, "seats": [s.to_dict() for s in self.seats]}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Transport":
        t = Transport(transport_id=d["transport_id"], model=d["model"], capacity=d["capacity"], seats=[Seat.from_dict(s) for s in d.get("seats", [])])
        return t


class Bus(Transport):
    pass


class Train(Transport):
    pass


class Plane(Transport):
    pass


@dataclass
class Route:
    route_id: str
    origin: str
    destination: str
    stops: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Route":
        return Route(route_id=d["route_id"], origin=d["origin"], destination=d["destination"], stops=d.get("stops", []))


@dataclass
class Trip:
    trip_id: str
    route: Route
    transport: Transport
    departure: datetime
    arrival: datetime
    base_price: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trip_id": self.trip_id,
            "route": self.route.to_dict(),
            "transport": self.transport.to_dict(),
            "departure": self.departure.isoformat(),
            "arrival": self.arrival.isoformat(),
            "base_price": self.base_price
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Trip":
        route = Route.from_dict(d["route"])
        transport = Transport.from_dict(d["transport"])
        departure = datetime.fromisoformat(d["departure"])
        arrival = datetime.fromisoformat(d["arrival"])
        return Trip(trip_id=d["trip_id"], route=route, transport=transport, departure=departure, arrival=arrival, base_price=d.get("base_price", 0.0))


# Ticket, Booking, Payment
@dataclass
class Ticket:
    ticket_id: str
    passenger: Passenger
    trip_id: str
    seat_id: str
    price: float

    def to_dict(self) -> Dict[str, Any]:
        return {"ticket_id": self.ticket_id, "passenger": self.passenger.to_dict(), "trip_id": self.trip_id, "seat_id": self.seat_id, "price": self.price}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Ticket":
        passenger = Passenger.from_dict(d["passenger"])
        return Ticket(ticket_id=d["ticket_id"], passenger=passenger, trip_id=d["trip_id"], seat_id=d["seat_id"], price=d["price"])


@dataclass
class Payment:
    payment_id: str
    booking_id: str
    amount: float
    method: str  # предполагается "card", "cash"
    status: str = "pending"  # предполагается "pending", "paid", "failed"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Payment":
        return Payment(payment_id=d["payment_id"], booking_id=d["booking_id"], amount=d["amount"], method=d["method"], status=d.get("status", "pending"))


@dataclass
class Booking:
    booking_id: str
    user: User
    tickets: List[Ticket]
    status: str = "created"  # предполагается "created", "paid", "cancelled"
    created_at: datetime = field(default_factory=datetime.utcnow)

    def total_amount(self) -> float:
        return sum(t.price for t in self.tickets)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "booking_id": self.booking_id,
            "user": self.user.to_dict(),
            "tickets": [t.to_dict() for t in self.tickets],
            "status": self.status,
            "created_at": self.created_at.isoformat()
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Booking":
        user = User.from_dict(d["user"])
        tickets = [Ticket.from_dict(t) for t in d.get("tickets", [])]
        created_at = datetime.fromisoformat(d.get("created_at")) if d.get("created_at") else datetime.utcnow()
        return Booking(booking_id=d["booking_id"], user=user, tickets=tickets, status=d.get("status", "created"), created_at=created_at)


# BookingSystem — CRUD и сериализация
class BookingSystem:
    """Система хранения и CRUD операций для сущностей бронирования"""

    def __init__(self) -> None:
        self.users: Dict[str, User] = {}
        self.passengers: Dict[str, Passenger] = {}
        self.transports: Dict[str, Transport] = {}
        self.routes: Dict[str, Route] = {}
        self.trips: Dict[str, Trip] = {}
        self.bookings: Dict[str, Booking] = {}
        self.payments: Dict[str, Payment] = {}

    # User CRUD
    def create_user(self, name: str, contact: ContactInfo) -> User:
        user_id = str(uuid.uuid4())
        user = User(user_id=user_id, name=name, contact=contact)
        self.users[user_id] = user
        return user

    def get_user(self, user_id: str) -> User:
        try:
            return self.users[user_id]
        except KeyError:
            raise NotFoundError(f"User {user_id} not found")

    def update_user(self, user_id: str, name: Optional[str] = None, contact: Optional[ContactInfo] = None) -> User:
        user = self.get_user(user_id)
        if name:
            user.name = name
        if contact:
            user.contact = contact
        return user

    def delete_user(self, user_id: str) -> None:
        if user_id in self.users:
            del self.users[user_id]
        else:
            raise NotFoundError(f"User {user_id} not found")

    # Passenger CRUD
    def create_passenger(self, name: str, passport: Optional[str] = None,
                         contact: Optional[ContactInfo] = None) -> Passenger:
        pid = str(uuid.uuid4())
        p = Passenger(passenger_id=pid, name=name, passport=passport, contact=contact)
        self.passengers[pid] = p
        return p

    def get_passenger(self, pid: str) -> Passenger:
        try:
            return self.passengers[pid]
        except KeyError:
            raise NotFoundError(f"Passenger {pid} not found")

    def update_passenger(self, pid: str,
                         name: Optional[str] = None,
                         passport: Optional[str] = None,
                         contact: Optional[ContactInfo] = None) -> Passenger:
        if pid not in self.passengers:
            raise NotFoundError(f"Passenger {pid} not found")

        p = self.passengers[pid]

        if name is not None:
            p.name = name
        if passport is not None:
            p.passport = passport
        if contact is not None:
            p.contact = contact

        return p

    def delete_passenger(self, pid: str) -> None:
        if pid not in self.passengers:
            raise NotFoundError(f"Passenger {pid} not found")
        del self.passengers[pid]

    # Transport / Route / Trip CRUD
    def add_transport(self, transport: Transport) -> Transport:
        self.transports[transport.transport_id] = transport
        return transport

    def get_transport(self, transport_id: str) -> Transport:
        try:
            return self.transports[transport_id]
        except KeyError:
            raise NotFoundError(f"Transport {transport_id} not found")

    def update_transport(self, transport_id: str,
                         model: Optional[str] = None,
                         capacity: Optional[int] = None) -> Transport:
        if transport_id not in self.transports:
            raise NotFoundError(f"Transport {transport_id} not found")

        t = self.transports[transport_id]

        if model is not None:
            t.model = model
        if capacity is not None:
            t.capacity = capacity

        return t

    def delete_transport(self, transport_id: str) -> None:
        if transport_id not in self.transports:
            raise NotFoundError(f"Transport {transport_id} not found")
        del self.transports[transport_id]

    def add_route(self, route: Route) -> Route:
        self.routes[route.route_id] = route
        return route

    def get_route(self, route_id: str) -> Route:
        try:
            return self.routes[route_id]
        except KeyError:
            raise NotFoundError(f"Route {route_id} not found")

    def update_route(self, route_id: str,
                     origin: Optional[str] = None,
                     destination: Optional[str] = None,
                     distance_km: Optional[float] = None) -> Route:
        if route_id not in self.routes:
            raise NotFoundError(f"Route {route_id} not found")

        r = self.routes[route_id]

        if origin is not None:
            r.origin = origin
        if destination is not None:
            r.destination = destination
        if distance_km is not None:
            r.distance_km = distance_km

        return r

    def delete_route(self, route_id: str) -> None:
        if route_id not in self.routes:
            raise NotFoundError(f"Route {route_id} not found")
        del self.routes[route_id]

    def add_trip(self, trip: Trip) -> Trip:
        if trip.trip_id in self.trips:
            raise ValidationError(f"Trip {trip.trip_id} already exists")
        self.trips[trip.trip_id] = trip
        return trip

    def get_trip(self, trip_id: str) -> Trip:
        try:
            return self.trips[trip_id]
        except KeyError:
            raise NotFoundError(f"Trip {trip_id} not found")

    def update_trip(self, trip_id: str,
                    date: Optional[str] = None,
                    price: Optional[float] = None,
                    route: Optional[Route] = None,
                    transport: Optional[Transport] = None) -> Trip:
        if trip_id not in self.trips:
            raise NotFoundError(f"Trip {trip_id} not found")

        tr = self.trips[trip_id]

        if date is not None:
            tr.date = date
        if price is not None:
            tr.price = price
        if route is not None:
            tr.route = route
        if transport is not None:
            tr.transport = transport

        return tr

    def delete_trip(self, trip_id: str) -> None:
        if trip_id not in self.trips:
            raise NotFoundError(f"Trip {trip_id} not found")
        del self.trips[trip_id]

    # --- Booking operations ---
    def create_booking(self, user_id: str, ticket_requests: List[Dict[str, Any]]) -> Booking:
        """ ticket_requests: list of dicts: {"passenger_id":..., "trip_id":..., "seat_id":..., "price":...} """
        user = self.get_user(user_id)
        tickets: List[Ticket] = []
        reserved_pairs: List[tuple] = []
        try:
            for req in ticket_requests:
                passenger = self.get_passenger(req["passenger_id"])
                trip = self.get_trip(req["trip_id"])
                seat_id = req["seat_id"]
                trip.transport.reserve_seat(seat_id)
                reserved_pairs.append((trip.transport, seat_id))
                ticket_id = str(uuid.uuid4())
                price = float(req.get("price", trip.base_price))
                ticket = Ticket(ticket_id=ticket_id, passenger=passenger, trip_id=trip.trip_id, seat_id=seat_id, price=price)
                tickets.append(ticket)
        except BookingError:
            for transport, sid in reserved_pairs:
                try:
                    transport.release_seat(sid)
                except Exception:
                    pass
            raise

        booking_id = str(uuid.uuid4())
        booking = Booking(booking_id=booking_id, user=user, tickets=tickets)
        self.bookings[booking_id] = booking
        return booking

    def get_booking(self, booking_id: str) -> Booking:
        try:
            return self.bookings[booking_id]
        except KeyError:
            raise NotFoundError(f"Booking {booking_id} not found")

    def cancel_booking(self, booking_id: str) -> None:
        booking = self.get_booking(booking_id)
        if booking.status == "cancelled":
            return
        for t in booking.tickets:
            trip = self.get_trip(t.trip_id)
            try:
                trip.transport.release_seat(t.seat_id)
            except NotFoundError:
                pass
        booking.status = "cancelled"

    # Payment simulation
    def pay_booking(self, booking_id: str, method: str = "card") -> Payment:
        booking = self.get_booking(booking_id)
        if booking.status == "cancelled":
            raise PaymentError("Cannot pay for cancelled booking")
        amount = booking.total_amount()
        payment_id = str(uuid.uuid4())
        payment = Payment(payment_id=payment_id, booking_id=booking_id, amount=amount, method=method, status="paid" if amount < 10000 else "failed")
        if payment.status != "paid":
            self.payments[payment_id] = payment
            raise PaymentError("Payment failed (simulated)")
        booking.status = "paid"
        self.payments[payment_id] = payment
        return payment

    # Serialization helpers
    def to_dict(self) -> Dict[str, Any]:
        return {
            "users": {uid: u.to_dict() for uid, u in self.users.items()},
            "passengers": {pid: p.to_dict() for pid, p in self.passengers.items()},
            "transports": {tid: t.to_dict() for tid, t in self.transports.items()},
            "routes": {rid: r.to_dict() for rid, r in self.routes.items()},
            "trips": {tid: tr.to_dict() for tid, tr in self.trips.items()},
            "bookings": {bid: b.to_dict() for bid, b in self.bookings.items()},
            "payments": {pid: p.to_dict() for pid, p in self.payments.items()}
        }

    def save_to_json(self, filename: str) -> None:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @staticmethod
    def load_from_json(filename: str) -> "BookingSystem":
        bs = BookingSystem()
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Load users
        for uid, ud in data.get("users", {}).items():
            bs.users[uid] = User.from_dict(ud)
        for pid, pd in data.get("passengers", {}).items():
            bs.passengers[pid] = Passenger.from_dict(pd)
        for tid, td in data.get("transports", {}).items():
            bs.transports[tid] = Transport.from_dict(td)
        for rid, rd in data.get("routes", {}).items():
            bs.routes[rid] = Route.from_dict(rd)
        for trid, trd in data.get("trips", {}).items():
            bs.trips[trid] = Trip.from_dict(trd)
        for bid, bd in data.get("bookings", {}).items():
            bs.bookings[bid] = Booking.from_dict(bd)
        for pid, pd in data.get("payments", {}).items():
            bs.payments[pid] = Payment.from_dict(pd)
        return bs

    def save_to_xml(self, filename: str) -> None:
        root = ET.Element("BookingSystem")
        def map_to_xml(parent: ET.Element, name: str, mapping: Dict[str, Any]) -> None:
            container = ET.SubElement(parent, name)
            for key, val in mapping.items():
                item = ET.SubElement(container, "item", id=key)
                item.text = json.dumps(val, ensure_ascii=False)

        map_to_xml(root, "users", {k: v.to_dict() for k, v in self.users.items()})
        map_to_xml(root, "passengers", {k: v.to_dict() for k, v in self.passengers.items()})
        map_to_xml(root, "transports", {k: v.to_dict() for k, v in self.transports.items()})
        map_to_xml(root, "routes", {k: v.to_dict() for k, v in self.routes.items()})
        map_to_xml(root, "trips", {k: v.to_dict() for k, v in self.trips.items()})
        map_to_xml(root, "bookings", {k: v.to_dict() for k, v in self.bookings.items()})
        map_to_xml(root, "payments", {k: v.to_dict() for k, v in self.payments.items()})

        tree = ET.ElementTree(root)
        tree.write(filename, encoding="utf-8", xml_declaration=True)

    @staticmethod
    def load_from_xml(filename: str) -> "BookingSystem":
        bs = BookingSystem()
        tree = ET.parse(filename)
        root = tree.getroot()
        for container in root:
            name = container.tag
            for item in container.findall("item"):
                key = item.attrib.get("id")
                text = item.text or "{}"
                obj = json.loads(text)
                if name == "users":
                    bs.users[key] = User.from_dict(obj)
                elif name == "passengers":
                    bs.passengers[key] = Passenger.from_dict(obj)
                elif name == "transports":
                    bs.transports[key] = Transport.from_dict(obj)
                elif name == "routes":
                    bs.routes[key] = Route.from_dict(obj)
                elif name == "trips":
                    bs.trips[key] = Trip.from_dict(obj)
                elif name == "bookings":
                    bs.bookings[key] = Booking.from_dict(obj)
                elif name == "payments":
                    bs.payments[key] = Payment.from_dict(obj)
        return bs


# Демонстрация
def _demo() -> None:
    system = BookingSystem()

    # Создадим пользователя и пассажира
    user = system.create_user("Ivan Petrov", ContactInfo(phone="+7-900-111-22-33", email="ivan@example.com"))
    passenger = system.create_passenger("Ivan Petrov", passport="AA1234567", contact=ContactInfo(phone="+7-900-111-22-33"))

    # Транспорт (например, автобус с 5 мест)
    bus = Bus(transport_id="BUS-001", model="Volvo B10", capacity=5)
    system.add_transport(bus)

    # Маршрут и трип
    route = Route(route_id="R-100", origin="Moscow", destination="Tula", stops=["Serpuhov"])
    system.add_route(route)
    trip = Trip(trip_id="TRIP-1000", route=route, transport=bus, departure=datetime(2025, 12, 1, 9, 0), arrival=datetime(2025, 12, 1, 12, 0), base_price=500.0)
    system.add_trip(trip)

    # Выберем доступное место
    available = bus.get_available_seats()
    if not available:
        print("Нет свободных мест")
        return
    chosen_seat = available[0].seat_id

    # Создание бронирования
    try:
        booking = system.create_booking(user.user_id, [{"passenger_id": passenger.passenger_id, "trip_id": trip.trip_id, "seat_id": chosen_seat, "price": 550.0}])
        print("Booking created:", booking.booking_id, "Total:", booking.total_amount())
    except BookingError as e:
        print("Booking failed:", e)
        return

    # Попытка оплатить
    try:
        payment = system.pay_booking(booking.booking_id, method="card")
        print("Payment succeeded:", payment.payment_id)
    except PaymentError as e:
        print("Payment failed:", e)

    # Сохраним в JSON и XML
    system.save_to_json("data.json")
    system.save_to_xml("data.xml")
    print("Saved to data.json and data.xml")

    # Продемонстрируем загрузку
    loaded = BookingSystem.load_from_json("data.json")
    print("Loaded bookings:", list(loaded.bookings.keys()))


if __name__ == "__main__":
    _demo()
